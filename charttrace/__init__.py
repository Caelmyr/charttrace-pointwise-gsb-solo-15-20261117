import io
from flask import jsonify,render_template,request,send_file,redirect
from image_lab.validation import Invalid
from .store import Store,decode,encode
from .delivery import demo,demo_bars,extract,csv_bytes,bundle
from .bars import bars_csv_bytes


def install(app,root):
    store=Store(root)
    app.extensions['charttrace']=store

    @app.get('/')
    def chart_home():
        return render_template('chart.html')

    @app.get('/chart/api/projects')
    def listing():
        return jsonify(store.listing())

    @app.post('/chart/api/projects')
    def create():
        body=request.get_json(silent=True) or {}
        if body.get('kind')=='bars':
            image,calibration,bars=demo_bars()
            p=store.create(body.get('name','堆叠柱图表示范'),image)
            return jsonify(store.update(p['id'],p['revision'],{'calibration':calibration,'bars':bars}))
        p=store.create(body.get('name','合成图表示范'),demo())
        p=store.update(p['id'],p['revision'],{'calibration':{'origin':[100,460],'x_end':[760,460],
            'y_end':[100,65],'x':{'min':0,'max':10,'scale':'linear'},
            'y':{'min':0,'max':100,'scale':'linear'}}})
        return jsonify(p)

    @app.get('/chart/api/projects/<key>')
    def get_project(key):
        return jsonify(store.read(key))

    @app.patch('/chart/api/projects/<key>')
    def update(key):
        data=request.get_json()
        return jsonify(store.update(key,data['revision'],data))

    @app.post('/chart/api/projects/<key>/source')
    def upload(key):
        return jsonify(store.replace_source(key,int(request.form['revision']),decode(request.files['file'].read())))

    @app.get('/chart/api/projects/<key>/source.png')
    def source(key):
        return send_file(io.BytesIO(encode(store.source(store.read(key)))),mimetype='image/png')

    @app.post('/chart/api/projects/<key>/series')
    def add(key):
        d=request.get_json()
        return jsonify(store.add_series(key,d['revision'],d.get('name','新曲线')))

    @app.patch('/chart/api/projects/<key>/series/<sid>')
    def edit(key,sid):
        d=request.get_json()
        return jsonify(store.edit_series(key,d['revision'],sid,d))

    @app.delete('/chart/api/projects/<key>/series/<sid>')
    def remove(key,sid):
        d=request.get_json()
        return jsonify(store.delete_series(key,d['revision'],sid))

    @app.post('/chart/api/projects/<key>/versions')
    def snapshot(key):
        d=request.get_json()
        return jsonify(store.snapshot(key,d['revision'],d.get('label','命名版本')))

    @app.post('/chart/api/projects/<key>/versions/<vid>/restore')
    def restore(key,vid):
        d=request.get_json()
        return jsonify(store.restore(key,d['revision'],vid))

    @app.post('/chart/api/projects/<key>/extract')
    def run(key):
        p,image=store.capture(key)
        series,bars,overlay=extract(p,image)
        return jsonify(store.publish(p,series,overlay,bars))

    @app.get('/chart/api/projects/<key>/results/<rid>')
    def result(key,rid):
        record,folder=store.result(key,None if rid=='current' else rid)
        return jsonify(record)

    @app.get('/chart/api/projects/<key>/results/<rid>/<artifact>')
    def download(key,rid,artifact):
        record,folder=store.result(key,None if rid=='current' else rid)
        if artifact=='overlay.png':
            return send_file(folder/'overlay.png',mimetype='image/png')
        if artifact=='data.csv':
            return send_file(io.BytesIO(csv_bytes(record['series'])),mimetype='text/csv',as_attachment=True,download_name='chart-data.csv')
        if artifact=='bars.csv':
            if not record.get('bars'):
                raise Invalid('该结果没有堆叠柱数据')
            return send_file(io.BytesIO(bars_csv_bytes(record['bars'])),mimetype='text/csv',as_attachment=True,download_name='chart-bars.csv')
        if artifact=='bundle.zip':
            return send_file(io.BytesIO(bundle(record,folder)),mimetype='application/zip',as_attachment=True,download_name='chart-extraction.zip')
        raise Invalid('文件不存在')
