import io
from flask import jsonify,render_template,request,send_file
from image_lab.validation import Invalid
from .delivery import demo,demo_panels,extract,csv_bytes,bundle,flatten
from . import panels as panel_model


def install(app,root):
    from .store import Store,decode,encode
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
        data=request.get_json(silent=True) or {}
        multi=bool(data.get('panels'))
        p=store.create(data.get('name','多面板示范' if multi else '合成图表示范'),
                       demo_panels() if multi else demo(),multi=multi)
        return jsonify(store.view(p['id']))

    @app.get('/chart/api/projects/<key>')
    def get_project(key):
        return jsonify(store.view(key))

    @app.patch('/chart/api/projects/<key>')
    def update(key):
        data=request.get_json()
        return jsonify(store.view(store.update(key,data['revision'],data)['id']))

    @app.post('/chart/api/projects/<key>/source')
    def upload(key):
        return jsonify(store.view(store.replace_source(key,int(request.form['revision']),
                                                       decode(request.files['file'].read()))['id']))

    @app.get('/chart/api/projects/<key>/source.png')
    def source(key):
        return send_file(io.BytesIO(encode(store.source(store.read(key)))),mimetype='image/png')

    @app.post('/chart/api/projects/<key>/panels')
    def add_panel(key):
        d=request.get_json()
        return jsonify(store.view(store.add_panel(key,d['revision'],d)['id']))

    @app.patch('/chart/api/projects/<key>/panels/<pid>')
    def edit_panel(key,pid):
        d=request.get_json()
        return jsonify(store.view(store.edit_panel(key,d['revision'],pid,d)['id']))

    @app.delete('/chart/api/projects/<key>/panels/<pid>')
    def delete_panel(key,pid):
        d=request.get_json()
        return jsonify(store.view(store.delete_panel(key,d['revision'],pid)['id']))

    @app.post('/chart/api/projects/<key>/series')
    def add(key):
        d=request.get_json()
        return jsonify(store.view(store.add_series(key,d['revision'],d.get('name','新曲线'),
                                                   d.get('panel'))['id']))

    @app.patch('/chart/api/projects/<key>/series/<sid>')
    def edit(key,sid):
        d=request.get_json()
        return jsonify(store.view(store.edit_series(key,d['revision'],sid,d,d.get('panel'))['id']))

    @app.delete('/chart/api/projects/<key>/series/<sid>')
    def remove(key,sid):
        d=request.get_json()
        return jsonify(store.view(store.delete_series(key,d['revision'],sid,d.get('panel'))['id']))

    @app.post('/chart/api/projects/<key>/versions')
    def snapshot(key):
        d=request.get_json()
        return jsonify(store.view(store.snapshot(key,d['revision'],d.get('label','命名版本'))['id']))

    @app.post('/chart/api/projects/<key>/versions/<vid>/restore')
    def restore(key,vid):
        d=request.get_json()
        return jsonify(store.view(store.restore(key,d['revision'],vid)['id']))

    @app.post('/chart/api/projects/<key>/extract')
    def run(key):
        p,image=store.capture(key)
        panels,overlay=extract(p,image)
        return jsonify(store.publish(p,panels,overlay))

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
            return send_file(io.BytesIO(csv_bytes(flatten(record['panels']))),mimetype='text/csv',
                             as_attachment=True,download_name='chart-data.csv')
        if artifact=='bundle.zip':
            return send_file(io.BytesIO(bundle(record,folder)),mimetype='application/zip',
                             as_attachment=True,download_name='chart-extraction.zip')
        raise Invalid('文件不存在')
