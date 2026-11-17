"""Portable ChartTrace packages that keep the source image, calibration and
proofreading state together so work can continue on another installation.

A package is a ZIP with a small manifest (``chartedit.json``), content-addressed
source images under ``sources/`` and one folder per captured result.  Source
images are stored once per SHA-256, so several named versions based on the same
upload do not duplicate the image.

The package never rewrites an old result's state from the project's current
settings: a result is exported from its own stored capture, named versions from
their own stored snapshot, and the working draft only when explicitly chosen.
Import always creates a fresh project with a fresh id and a non-clashing name.
"""
import csv
import hashlib
import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from image_lab.validation import Invalid
from .coordinates import validate as validate_calibration
from .store import identifier, settings as normalize_settings, decode

FORMAT = 'charttrace-editable'
FORMAT_VERSION = 1
MAX_PACKAGE_BYTES = 96 * 1024 * 1024
MAX_UNCOMPRESSED = 512 * 1024 * 1024
MAX_ENTRIES = 256
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_ENTRY_BYTES = 128 * 1024 * 1024
MAX_DOCUMENTS = 25
MAX_RESULT_ROWS = 1000000
MAX_PATH_POINTS = 10000
SHA = re.compile('[0-9a-f]{64}')

RESULT_KINDS = ('result', 'working', 'version')


# ---------------------------------------------------------------------------
# Export


def _stored_source(store, project, filename, context):
    name = str(filename or '')
    if not name or Path(name).name != name or name in ('.', '..'):
        raise Invalid(context + '的原图引用无效，无法导出')
    path = store.directory(project['id']) / name
    if not path.is_file():
        raise Invalid(context + '缺少原图，无法导出')
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Invalid(context + '的原图无法读取，无法导出') from error
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise Invalid(context + '的原图过大或为空，无法导出')
    try:
        decode(raw)
    except Invalid as error:
        raise Invalid(context + '的原图无法读取：' + str(error)) from error
    return raw


def _export_series(series, context):
    if not isinstance(series, list) or len(series) > 8:
        raise Invalid(context + '的曲线状态无效，无法导出')
    result = []
    for item in series:
        if not isinstance(item, dict) or not isinstance(item.get('settings'), dict):
            raise Invalid(context + '的曲线状态无效，无法导出')
        anchors = []
        for anchor in item.get('anchors') or []:
            try:
                pixel = np.asarray(anchor['pixel'], dtype=float)
            except (KeyError, TypeError, ValueError):
                raise Invalid(context + '的手动校对点已损坏，无法导出') from None
            if pixel.shape != (2,) or not np.isfinite(pixel).all():
                raise Invalid(context + '的手动校对点已损坏，无法导出')
            anchors.append({'id': str(anchor.get('id', '')), 'pixel': pixel.tolist()})
        name = item.get('name', '曲线')
        if not isinstance(name, str):
            raise Invalid(context + '的曲线名称无效，无法导出')
        result.append({'id': str(item.get('id', '')), 'name': name[:80],
                       'settings': dict(item['settings']), 'anchors': anchors})
    return result


def _captured_document(store, project, document, context):
    if not isinstance(document, dict):
        raise Invalid(context + '缺少可编辑状态，无法导出')
    try:
        calibration = validate_calibration(document.get('calibration'))
    except Invalid as error:
        raise Invalid(context + '的标定无效，无法导出：' + str(error)) from None
    series = _export_series(document.get('series'), context)
    raw = _stored_source(store, project, document.get('source'), context)
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'raw': raw,
            'calibration': calibration, 'series': series}


def _csv(series):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['series', 'segment', 'x', 'y', 'pixel_x', 'pixel_y', 'confidence'])
    for s in series:
        for row in s['rows']:
            writer.writerow([s['name']] + [row[k] for k in
                                           ['segment', 'x', 'y', 'pixel_x', 'pixel_y', 'confidence']])
    return output.getvalue().encode('utf-8-sig')


def build_package(store, key, result_id='', version_ids=()):
    """Build an editable ZIP for one state plus optional named versions."""
    if not isinstance(version_ids, (list, tuple)):
        raise Invalid('命名版本选择无效')
    if len(version_ids) > 20:
        raise Invalid('一次最多带走20个命名版本')
    with store.lock:
        p = store.read(key)
        sources = {}
        records = {}
        documents = []

        def register(document):
            sha = document['sha256']
            sources.setdefault(sha, document)
            return sha

        if result_id in ('', 'working', None):
            # The working draft is only exported when it is explicitly chosen;
            # a captured result below always uses its own frozen document.
            if p.get('source') is None:
                raise Invalid('工程还没有原图，无法导出')
            document = _captured_document(store, p, {
                'source': p.get('source'), 'calibration': p.get('calibration'),
                'series': p.get('series')}, '当前编辑状态')
            primary = {'key': 'working', 'kind': 'working', 'label': '当前编辑状态',
                       'source': register(document), 'calibration': document['calibration'],
                       'series': document['series']}
            documents.append(primary)
        else:
            record, folder = store.result(p['id'], None if result_id == 'current' else result_id)
            context = '结果 REV ' + str(record['revision'])
            document = _captured_document(store, p, record.get('document'), context)
            if not isinstance(record.get('series'), list):
                raise Invalid(context + '的提取数据已损坏，无法导出')
            overlay_path = folder / 'overlay.png'
            if not overlay_path.is_file():
                raise Invalid(context + '缺少叠加图，无法导出')
            overlay = overlay_path.read_bytes()
            if not overlay.startswith(b'\x89PNG\r\n\x1a\n'):
                raise Invalid(context + '的叠加图已损坏，无法导出')
            primary = {'key': 'result:' + record['id'], 'kind': 'result',
                       'label': '提取结果 REV ' + str(record['revision']),
                       'source': register(document), 'calibration': document['calibration'],
                       'series': document['series'],
                       'result': {'id': record['id'], 'revision': record['revision'],
                                  'created': record['created'],
                                  'files': {'result': 'results/' + record['id'] + '/result.json',
                                            'overlay': 'results/' + record['id'] + '/overlay.png',
                                            'csv': 'results/' + record['id'] + '/data.csv'}}}
            documents.append(primary)
            packaged = {'id': record['id'], 'revision': record['revision'],
                        'created': record['created'],
                        'document': {'source': document['sha256'],
                                     'calibration': document['calibration'],
                                     'series': document['series']},
                        'series': record['series']}
            records[record['id']] = {'record': packaged, 'overlay': overlay,
                                     'csv': _csv(record['series'])}

        seen = set()
        for version_id in version_ids:
            if not isinstance(version_id, str) or not version_id or version_id in seen:
                continue
            seen.add(version_id)
            version = next((v for v in p.get('versions', []) if v.get('id') == version_id), None)
            if version is None:
                raise Invalid('命名版本不存在或已删除')
            label = version.get('label', '命名版本')
            label = label[:80] if isinstance(label, str) and label else '命名版本'
            document = _captured_document(store, p, version.get('document'),
                                          '命名版本“' + label + '”')
            documents.append({'key': 'version:' + version_id, 'kind': 'version', 'label': label,
                              'source': register(document),
                              'calibration': document['calibration'],
                              'series': document['series']})

        manifest = {
            'format': FORMAT,
            'format_version': FORMAT_VERSION,
            'created': time.time(),
            'application': 'ChartTrace',
            'project': {'name': p['name']},
            'primary': primary['key'],
            'sources': [{'sha256': sha, 'path': 'sources/' + sha + '.png',
                         'bytes': len(sources[sha]['raw'])} for sha in sources],
            'documents': documents,
        }
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('chartedit.json', json.dumps(manifest, ensure_ascii=False, indent=2))
            for sha, document in sources.items():
                archive.writestr('sources/' + sha + '.png', document['raw'])
            for rid, item in records.items():
                folder = 'results/' + rid
                archive.writestr(folder + '/result.json',
                                 json.dumps(item['record'], ensure_ascii=False, indent=2))
                archive.writestr(folder + '/overlay.png', item['overlay'])
                archive.writestr(folder + '/data.csv', item['csv'])
            primary_record = records.get((primary.get('result') or {}).get('id'))
            if primary_record is not None:
                # Keep the familiar single-result bundle contents at the root so
                # older tools can still open the download.
                archive.writestr('data.csv', primary_record['csv'])
                archive.writestr('extraction.json',
                                 json.dumps(primary_record['record'], ensure_ascii=False, indent=2))
                archive.writestr('overlay.png', primary_record['overlay'])
        raw = output.getvalue()
        if len(raw) > MAX_PACKAGE_BYTES:
            raise Invalid('导出包超过96MB，请减少命名版本或先压缩原图')
        return raw


# ---------------------------------------------------------------------------
# Import


def _open_archive(raw):
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise Invalid('请选择ChartTrace可编辑包')
    if len(raw) > MAX_PACKAGE_BYTES:
        raise Invalid('可编辑包不得超过96MB')
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as error:
        raise Invalid('不是有效的ZIP可编辑包') from error
    try:
        entries = archive.infolist()
        if not entries:
            raise Invalid('可编辑包为空')
        if len(entries) > MAX_ENTRIES:
            raise Invalid('可编辑包文件过多，可能已损坏')
        names = set()
        total = 0
        for entry in entries:
            name = entry.filename
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
                    or ':' in name or name in names):
                raise Invalid('可编辑包包含无效或重复的文件路径')
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise Invalid('可编辑包不能包含符号链接')
            if entry.flag_bits & 1:
                raise Invalid('不支持加密的可编辑包')
            if entry.file_size > MAX_ENTRY_BYTES:
                raise Invalid('可编辑包内单个文件过大')
            total += entry.file_size
            if total > MAX_UNCOMPRESSED:
                raise Invalid('可编辑包解压尺寸超限')
            if entry.compress_size and entry.file_size / max(entry.compress_size, 1) > 3000:
                raise Invalid('可编辑包压缩率异常，可能已损坏')
            names.add(name)
        if 'chartedit.json' not in names:
            raise Invalid('不是ChartTrace可编辑包：缺少chartedit.json')
        return archive
    except Exception:
        archive.close()
        raise


def _read_json(archive, name, label):
    try:
        raw = archive.read(name)
    except KeyError as error:
        raise Invalid('可编辑包不完整：缺少' + label) from error
    if len(raw) > MAX_JSON_BYTES:
        raise Invalid('可编辑包中的' + label + '过大')
    try:
        return json.loads(raw.decode('utf8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Invalid('可编辑包中的' + label + '已损坏') from error


def _finite(value, label):
    if isinstance(value, bool):
        raise Invalid(label + '必须是数字')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise Invalid(label + '必须是数字') from None
    if not np.isfinite(number):
        raise Invalid(label + '必须是有限数字')
    return number


def _read_anchors(value, label):
    if value is None:
        return []
    if not isinstance(value, list):
        raise Invalid(label + '的手动校对点已损坏')
    if len(value) > 200:
        raise Invalid(label + '最多200个手动校对点')
    anchors = []
    for item in value:
        if not isinstance(item, dict):
            raise Invalid(label + '的手动校对点已损坏')
        try:
            point = np.asarray(item.get('pixel'), dtype=float)
        except (ValueError, TypeError):
            raise Invalid(label + '的手动校对点必须是像素坐标') from None
        if point.shape != (2,) or not np.isfinite(point).all():
            raise Invalid(label + '的手动校对点必须是两个有限像素坐标')
        anchors.append({'id': str(item.get('id', '')), 'pixel': point.tolist()})
    return anchors


def _read_document_series(value, label):
    if not isinstance(value, list):
        raise Invalid(label + '缺少曲线状态')
    if len(value) > 8:
        raise Invalid(label + '最多8条曲线')
    series = []
    for item in value:
        if not isinstance(item, dict):
            raise Invalid(label + '的曲线状态已损坏')
        name = item.get('name', '曲线')
        if not isinstance(name, str):
            raise Invalid(label + '的曲线名称无效')
        try:
            settings = normalize_settings(item.get('settings') or {})
        except (Invalid, ValueError, TypeError) as error:
            raise Invalid(label + '的曲线设置无效：' + str(error)) from None
        series.append({'id': str(item.get('id', '')), 'name': name[:80],
                       'settings': settings, 'anchors': _read_anchors(item.get('anchors'), label)})
    return series


def _read_result_rows(value, label):
    if not isinstance(value, list):
        raise Invalid(label + '的数据点已损坏')
    if len(value) > MAX_RESULT_ROWS:
        raise Invalid(label + '的数据点过多')
    rows = []
    for item in value:
        if not isinstance(item, dict):
            raise Invalid(label + '的数据点已损坏')
        segment = item.get('segment')
        if isinstance(segment, bool) or not isinstance(segment, int) or segment < 0:
            raise Invalid(label + '的数据片段编号无效')
        row = {'segment': segment}
        for key in ('x', 'y', 'pixel_x', 'pixel_y', 'confidence'):
            row[key] = _finite(item.get(key), label + '的' + key)
        rows.append(row)
    return rows


def _read_paths(value, label):
    if not isinstance(value, list):
        raise Invalid(label + '的追踪片段已损坏')
    if len(value) > 4000:
        raise Invalid(label + '的追踪片段过多')
    paths = []
    for item in value:
        if not isinstance(item, dict):
            raise Invalid(label + '的追踪片段已损坏')
        chart = item.get('chart')
        observed = item.get('observed')
        confidence = item.get('confidence')
        if (not isinstance(chart, list) or not isinstance(observed, list)
                or not isinstance(confidence, list)
                or len(chart) > MAX_PATH_POINTS or len(observed) != len(chart)
                or len(confidence) != len(chart)):
            raise Invalid(label + '的追踪片段已损坏')
        points = []
        for point in chart:
            try:
                pair = np.asarray(point, dtype=float)
            except (ValueError, TypeError):
                raise Invalid(label + '的追踪片段坐标已损坏') from None
            if pair.shape != (2,) or not np.isfinite(pair).all():
                raise Invalid(label + '的追踪片段坐标已损坏')
            points.append(pair.tolist())
        for flag in observed:
            if not isinstance(flag, bool):
                raise Invalid(label + '的追踪片段标记已损坏')
        paths.append({'chart': points, 'observed': observed,
                      'confidence': [_finite(v, label + '的置信度') for v in confidence]})
    return paths


def _read_result_series(value, label):
    if not isinstance(value, list):
        raise Invalid(label + '的提取数据已损坏')
    if len(value) > 8:
        raise Invalid(label + '最多8条曲线')
    series = []
    for item in value:
        if not isinstance(item, dict):
            raise Invalid(label + '的曲线结果已损坏')
        name = item.get('name', '曲线')
        if not isinstance(name, str):
            raise Invalid(label + '的曲线名称无效')
        series.append({'id': str(item.get('id', '')), 'name': name[:80],
                       'paths': _read_paths(item.get('paths'), label),
                       'rows': _read_result_rows(item.get('rows'), label),
                       'observed_columns': int(_finite(item.get('observed_columns', 0),
                                                      label + '的观测列数')),
                       'mean_confidence': _finite(item.get('mean_confidence', 0),
                                                  label + '的平均置信度')})
    return series


def _identifier(value, label):
    if not isinstance(value, str) or len(value) != 24 or any(c not in '0123456789abcdef' for c in value):
        raise Invalid(label + '无效')
    return value


def _read_result(archive, document, value, label):
    if not isinstance(value, dict):
        raise Invalid(label + '缺少结果信息，可编辑包不完整')
    rid = _identifier(value.get('id'), label + '的结果编号')
    base = 'results/' + rid
    files = value.get('files')
    expected = {'result': base + '/result.json', 'overlay': base + '/overlay.png',
                'csv': base + '/data.csv'}
    if (not isinstance(files, dict) or files.get('result') != expected['result']
            or files.get('overlay') != expected['overlay']
            or files.get('csv') not in (None, expected['csv'])):
        raise Invalid(label + '的结果文件引用无效')
    record = _read_json(archive, expected['result'], '结果数据')
    if not isinstance(record, dict) or record.get('id') != rid:
        raise Invalid(label + '的结果数据已损坏')
    revision = record.get('revision')
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise Invalid(label + '的结果修订号无效')
    created = _finite(record.get('created'), label + '的结果时间')
    captured = record.get('document')
    if not isinstance(captured, dict):
        raise Invalid(label + '的结果缺少当时的标定与校对状态')
    try:
        captured_calibration = validate_calibration(captured.get('calibration'))
    except Invalid as error:
        raise Invalid(label + '的结果标定已损坏：' + str(error)) from None
    try:
        captured_series = _read_document_series(captured.get('series'), label + '的结果')
    except Invalid as error:
        raise Invalid(label + '的结果校对状态已损坏：' + str(error)) from None
    if (captured.get('source') != document['source']
            or captured_calibration != document['calibration']
            or captured_series != document['series']):
        raise Invalid(label + '的结果与其可编辑状态不一致，可编辑包已损坏')
    rows = _read_result_series(record.get('series'), label)
    try:
        overlay = archive.read(expected['overlay'])
    except KeyError as error:
        raise Invalid('可编辑包不完整：' + label + '缺少叠加图') from error
    if not overlay.startswith(b'\x89PNG\r\n\x1a\n'):
        raise Invalid(label + '的叠加图已损坏')
    if len(overlay) > MAX_IMAGE_BYTES:
        raise Invalid(label + '的叠加图过大')
    try:
        overlay_image = decode(overlay)
    except Invalid as error:
        raise Invalid(label + '的叠加图无法读取：' + str(error)) from None
    if overlay_image.shape[:2] != document['shape'][:2]:
        raise Invalid(label + '的叠加图尺寸与原图不一致')
    return {'id': rid, 'revision': revision, 'created': created, 'series': rows,
            'overlay': overlay}


def read_package(raw):
    """Parse and validate an editable package without touching the store."""
    archive = _open_archive(raw)
    try:
        manifest = _read_json(archive, 'chartedit.json', '可编辑包清单')
        if not isinstance(manifest, dict) or manifest.get('format') != FORMAT:
            raise Invalid('不是ChartTrace可编辑包')
        if manifest.get('format_version') != FORMAT_VERSION:
            raise Invalid('不支持的可编辑包版本，请使用相同或更新版本的ChartTrace')
        project = manifest.get('project')
        name = project.get('name') if isinstance(project, dict) else None
        name = name.strip()[:100] if isinstance(name, str) and name.strip() else '导入的图表工程'

        entries = manifest.get('sources')
        if not isinstance(entries, list) or not entries:
            raise Invalid('可编辑包缺少原图，包不完整')
        sources = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise Invalid('可编辑包的原图清单已损坏')
            sha = entry.get('sha256')
            if not isinstance(sha, str) or not SHA.fullmatch(sha):
                raise Invalid('可编辑包的原图编号无效')
            if sha in sources or entry.get('path') != 'sources/' + sha + '.png':
                raise Invalid('可编辑包的原图路径无效或重复')
            path = 'sources/' + sha + '.png'
            try:
                payload = archive.read(path)
            except KeyError as error:
                raise Invalid('可编辑包缺少原图 ' + path + '，包不完整') from error
            if not payload or len(payload) > MAX_IMAGE_BYTES:
                raise Invalid('可编辑包中的原图为空或过大')
            if hashlib.sha256(payload).hexdigest() != sha:
                raise Invalid('可编辑包中的原图校验失败，文件可能已损坏')
            try:
                image = decode(payload)
            except Invalid as error:
                raise Invalid('可编辑包中的原图无法读取：' + str(error)) from None
            sources[sha] = {'raw': payload, 'shape': image.shape}
        if not sources:
            raise Invalid('可编辑包缺少原图，包不完整')

        values = manifest.get('documents')
        if not isinstance(values, list) or not values:
            raise Invalid('可编辑包缺少可编辑状态')
        if len(values) > MAX_DOCUMENTS:
            raise Invalid('可编辑包包含过多状态')
        documents = []
        keys = set()
        for entry in values:
            if not isinstance(entry, dict):
                raise Invalid('可编辑包的状态清单已损坏')
            key = entry.get('key')
            if not isinstance(key, str) or not key or len(key) > 80 or key in keys:
                raise Invalid('可编辑包的状态编号无效或重复')
            keys.add(key)
            kind = entry.get('kind')
            if kind not in RESULT_KINDS:
                raise Invalid('可编辑包包含不支持的状态类型')
            source = entry.get('source')
            if source not in sources:
                raise Invalid('可编辑包不完整：状态“' + key + '”缺少原图')
            label = entry.get('label')
            label = label[:80] if isinstance(label, str) and label else '未命名状态'
            try:
                calibration = validate_calibration(entry.get('calibration'))
            except Invalid as error:
                raise Invalid('可编辑包中“' + label + '”的标定无效：' + str(error)) from None
            series = _read_document_series(entry.get('series'), '可编辑包中“' + label + '”')
            document = {'key': key, 'kind': kind, 'label': label, 'source': source,
                        'shape': sources[source]['shape'], 'calibration': calibration,
                        'series': series, 'result': None}
            if kind == 'result':
                document['result'] = _read_result(archive, document, entry.get('result'),
                                                  '可编辑包中“' + label + '”')
            documents.append(document)

        primary = next((d for d in documents if d['key'] == manifest.get('primary')), None)
        if primary is None:
            raise Invalid('可编辑包缺少主编辑状态，包不完整')
        results = [d for d in documents if d['kind'] == 'result']
        if len(results) > 1:
            raise Invalid('可编辑包包含多个提取结果，当前版本无法导入')
        if results and results[0] is not primary:
            raise Invalid('可编辑包的主编辑状态与提取结果不一致')
        if primary['kind'] == 'result' and primary['result'] is None:
            raise Invalid('可编辑包的主结果不完整')
        if any(d['kind'] == 'working' for d in documents) and primary['kind'] != 'working':
            raise Invalid('可编辑包的主编辑状态不明确，包可能已损坏')
        return {'name': name, 'sources': sources, 'documents': documents, 'primary': primary}
    finally:
        archive.close()


def install_package(store, package):
    """Create a new project from a validated package; never overwrite an existing one."""
    with store.lock:
        name = store.unique_name(package['name'])
        project_id = identifier()
        while store.directory(project_id).exists():
            project_id = identifier()
        directory = store.directory(project_id)
        directory.mkdir(parents=True)
        try:
            source_names = {}
            for sha, source in package['sources'].items():
                filename = sha + '.png'
                (directory / filename).write_bytes(source['raw'])
                source_names[sha] = filename

            # Snapshots share curve ids; keep that relationship while giving the
            # imported project fresh anchor ids.
            def materialize(document):
                return {'source': source_names[document['source']],
                        'calibration': document['calibration'],
                        'series': [{'id': series['id'], 'name': series['name'],
                                    'settings': series['settings'],
                                    'anchors': [{'id': identifier(), 'pixel': a['pixel']}
                                                for a in series['anchors']]}
                                   for series in document['series']]}

            primary = package['primary']
            project = {'id': project_id, 'name': name, 'revision': 1, 'created': time.time(),
                       'source': None, 'calibration': primary['calibration'], 'series': [],
                       'versions': [], 'results': [], 'current_result': None}
            primary_document = materialize(primary)
            project.update(primary_document)

            for document in package['documents']:
                if document['kind'] != 'version':
                    continue
                project['versions'].append({'id': identifier(), 'label': document['label'],
                                            'document': materialize(document)})

            result = primary['result'] if primary['kind'] == 'result' else None
            if result is not None:
                rid = identifier()
                folder = directory / rid
                folder.mkdir()
                record = {'id': rid, 'revision': result['revision'], 'created': result['created'],
                          'document': primary_document, 'series': result['series']}
                (folder / 'result.json').write_text(json.dumps(record, ensure_ascii=False))
                (folder / 'overlay.png').write_bytes(result['overlay'])
                project['results'].append({'id': rid, 'revision': record['revision'],
                                           'created': record['created']})
                project['current_result'] = rid
            store.write(project)
            return project
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
