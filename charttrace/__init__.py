import io
from flask import jsonify,render_template,request,send_file
from image_lab.validation import Invalid
from .store import Store,decode,encode
from .delivery import demo,extract,csv_bytes,bundle
from .portable import build_package,read_package,install_package


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
        p=store.create((request.get_json(silent=True) or {}).get('name','合成图表示范'),demo())
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
        series,overlay=extract(p,image)
        return jsonify(store.publish(p,series,overlay))

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
        if artifact=='bundle.zip':
            return send_file(io.BytesIO(bundle(record,folder)),mimetype='application/zip',as_attachment=True,download_name='chart-extraction.zip')
        if artifact=='editable.zip':
            raw=build_package(store,key,None if rid=='current' else rid)
            return send_file(io.BytesIO(raw),mimetype='application/zip',as_attachment=True,
                             download_name='chart-editable.zip')
        raise Invalid('文件不存在')

    @app.post('/chart/api/projects/<key>/editable')
    def editable(key):
        data=request.get_json(silent=True) or {}
        if not isinstance(data,dict):
            raise Invalid('导出参数无效')
        result=data.get('result','current')
        if result is None:
            result='working'
        if not isinstance(result,str):
            raise Invalid('导出结果无效')
        raw=build_package(store,key,result,data.get('versions',[]))
        return send_file(io.BytesIO(raw),mimetype='application/zip',as_attachment=True,
                         download_name='chart-editable.zip')

    @app.post('/chart/api/import')
    def import_package():
        file=request.files.get('file')
        if file is None:
            raise Invalid('请选择可编辑包')
        package=read_package(file.read())
        return jsonify(install_package(store,package)),201
