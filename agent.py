import copy
import json
import re
from pathlib import Path

import numpy as np

from runtime import MODELS, read_json
from schemas import PRESETS, Simulation


def _number(text, pattern):
    match = re.search(pattern, text, re.I)
    return float(match.group(1)) if match else None


def _duration(text):
    matches=list(re.finditer(r'(?:仿真|计算|持续|加热|运行)?\s*(\d+(?:\.\d+)?)\s*(小时|h|分钟|min|秒|s)', text, re.I))
    if not matches:
        return None
    values=[]
    for match in matches:
        value=float(match.group(1));unit=match.group(2).lower()
        values.append(value*(3600 if unit in ('小时','h') else 60 if unit in ('分钟','min') else 1))
    duration=max(values)
    return duration if duration>0 else None


def _display_surface(model_id):
    display = read_json(MODELS / model_id / 'display.json')
    points = np.asarray(display['points'], dtype=float).reshape(-1, 3)
    faces = np.asarray(display['faces'], dtype=np.int64).reshape(-1, 3)
    tri = points[faces]
    centers = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-30)
    return centers, normals


def _face_selection(model_id, text, triangles):
    lower = text.lower()
    if re.search(r'整个外表面|全部外表面|所有表面|全表面', text, re.I):
        return list(range(triangles)), '已自动选择全部外表面'

    direction = None
    label = None
    for token, axis in (('+x', (0, 1)), ('-x', (0, -1)), ('+y', (1, 1)), ('-y', (1, -1)), ('+z', (2, 1)), ('-z', (2, -1))):
        if token in lower or re.search(rf'{axis[0] + 1}\s*轴\s*{("正" if axis[1] > 0 else "负")}', text):
            direction, label = axis, token
            break
    if direction is None:
        for pattern, axis, name in (
            (r'左(?:侧|边)?', (0, -1), '左侧 (-X)'),
            (r'右(?:侧|边)?', (0, 1), '右侧 (+X)'),
            (r'前(?:侧|面|方)', (1, -1), '前侧 (-Y)'),
            (r'后(?:侧|面|方)', (1, 1), '后侧 (+Y)'),
            (r'(?:底部|下方|下表面)', (2, -1), '底部 (-Z)'),
            (r'(?:顶部|上方|上表面)', (2, 1), '顶部 (+Z)'),
        ):
            if re.search(pattern, text):
                direction, label = axis, name
                break

    coordinate = re.search(r'\b([xyz])\s*(?:=|为|在)\s*(-?\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米|m|米)', text, re.I)
    if direction is None and coordinate:
        axis = 'xyz'.index(coordinate.group(1).lower())
        value = float(coordinate.group(2))
        unit = coordinate.group(3).lower()
        value *= .001 if unit in ('mm', '毫米') else .01 if unit in ('cm', '厘米') else 1
        centers, _ = _display_surface(model_id)
        distance = np.abs(centers[:, axis] - value)
        span = max(float(np.ptp(centers[:, axis])), 1e-12)
        if float(distance.min()) > span * .03:
            return None, None
        selected = np.flatnonzero(distance <= max(float(distance.min()) * 1.05, span * .03)).tolist()
        if not selected:
            selected = [int(distance.argmin())]
        return selected, f'已自动选择 {coordinate.group(1).upper()}={value:g} m 附近外表面 ({len(selected)} 个三角面)'
    if direction is None:
        return None, None

    centers, normals = _display_surface(model_id)
    axis, sign = direction
    edge = centers[:, axis].max() if sign > 0 else centers[:, axis].min()
    span = max(float(centers[:, axis].max() - centers[:, axis].min()), 1e-12)
    at_edge = np.abs(centers[:, axis] - edge) <= span * .03
    facing = normals[:, axis] * sign > .35
    selected = np.flatnonzero(at_edge & facing).tolist()
    if not selected:
        selected = np.flatnonzero(at_edge).tolist()
    return selected, f'已自动选择 {label} 外表面 ({len(selected)} 个三角面)'


def _position_from_prompt(model_id, text, faces=None):
    model=read_json(MODELS/model_id/'metadata.json')
    lo=np.asarray(model['bounds_m'][0],dtype=float);hi=np.asarray(model['bounds_m'][1],dtype=float)
    position=(lo+hi)/2
    found=False
    for match in re.finditer(r'\b([xyz])\s*(?:=|为|在)\s*(-?\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米|m|米)',text,re.I):
        axis='xyz'.index(match.group(1).lower());value=float(match.group(2));unit=match.group(3).lower()
        position[axis]=value*(.001 if unit in ('mm','毫米') else .01 if unit in ('cm','厘米') else 1);found=True
    if found:return position.tolist()
    if re.search(r'中心|中部|内部中心',text):return position.tolist()
    if faces:
        centers,_=_display_surface(model_id)
        return centers[np.asarray(faces,dtype=int)].mean(axis=0).tolist()
    return None


def make_plan(model_id, prompt, current):
    model = read_json(MODELS / model_id / 'metadata.json')
    cfg = copy.deepcopy(current or {})
    cfg['model_id'] = model_id
    changes, warnings, questions = [], [], []

    material = None
    aliases = (
        ('不锈钢|stainless', '不锈钢（示例）'),
        ('碳钢|steel', '碳钢（示例）'),
        ('铝|alum|aluminium', '铝（示例）'),
        ('铜|copper', '铜 C11000'),
        ('铁|iron', '铁（示例）'),
    )
    for pattern, name in aliases:
        if re.search(pattern, prompt, re.I):
            material = next(item for item in PRESETS if item['name'] == name)
            break
    if material:
        cfg['base_material'] = copy.deepcopy(material)
        changes.append(f'基础材料: {material["name"]}')

    if re.search(r'稳态|steady|最终稳定', prompt, re.I):
        cfg['analysis_mode'] = 'steady'
        changes.append('分析类型: 稳态导热')
    elif re.search(r'瞬态|transient|升温|降温|冷却过程', prompt, re.I):
        cfg['analysis_mode'] = 'transient'
        changes.append('分析类型: 瞬态导热')

    source_type='point' if re.search(r'点热源|点源|point',prompt,re.I) else 'surface'
    placement='embedded' if re.search(r'嵌入|内部|embedded',prompt,re.I) else 'external' if re.search(r'外部|外置|external',prompt,re.I) else 'surface'

    power = _number(prompt, r'(?:功率|加热|热源)?\s*(\d+(?:\.\d+)?)\s*(kW|千瓦|W|瓦)')
    if power is not None:
        unit = re.search(r'(kW|千瓦|W|瓦)', prompt[re.search(r'(?:功率|加热|热源)?\s*\d+(?:\.\d+)?\s*(kW|千瓦|W|瓦)', prompt, re.I).start():], re.I).group(1).lower()
        power *= 1000 if unit in ('kw', '千瓦') else 1
        cfg.setdefault('heat_sources', [])
        source = copy.deepcopy(cfg['heat_sources'][0]) if cfg['heat_sources'] else dict(name='智能热源', power_W=power, start_s=0, end_s=cfg.get('duration_s', 3600), faces=[])
        source.update(source_type=source_type, placement=placement, power_W=power, end_s=cfg.get('duration_s', 3600))
        cfg['heat_sources'] = [source] + cfg['heat_sources'][1:]
        changes.append(f'热源功率: {power:g} W')

    duration = _duration(prompt)
    if duration is not None:
        cfg['duration_s'] = duration
        for source in cfg.get('heat_sources', []):
            if source.get('end_s', 0) >= cfg.get('duration_s', duration):
                source['end_s'] = duration
        changes.append(f'仿真时长: {duration:g} s')

    for key, label, pattern in (
        ('initial_C', '初始温度', r'(?:初始|起始)温度?\s*(?:为|=|:)?\s*(-?\d+(?:\.\d+)?)\s*°?C'),
        ('ambient_C', '环境温度', r'(?:环境|室温)温度?\s*(?:为|=|:)?\s*(-?\d+(?:\.\d+)?)\s*°?C'),
        ('default_h', '换热系数', r'(?:换热系数|对流系数|h)\s*(?:为|=|:)?\s*(\d+(?:\.\d+)?)'),
        ('mesh_size_m', '网格尺寸', r'(?:网格(?:尺寸|大小)?|mesh)\s*(?:为|=|:)?\s*(\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米|m|米)'),
        ('dt_s', '计算步长', r'(?:计算|时间)?步长\s*(?:为|=|:)?\s*(\d+(?:\.\d+)?)\s*(秒|s)'),
        ('save_s', '保存间隔', r'(?:保存|输出)(?:间隔)?\s*(?:为|=|:)?\s*(\d+(?:\.\d+)?)\s*(秒|s)'),
    ):
        value = _number(prompt, pattern)
        if value is None:
            continue
        match = re.search(pattern, prompt, re.I)
        unit = match.group(2).lower() if match and match.lastindex and match.lastindex >= 2 else ''
        if key == 'mesh_size_m':
            value *= .001 if unit in ('mm', '毫米') else .01 if unit in ('cm', '厘米') else 1
        cfg[key] = value
        changes.append(f'{label}: {value:g} ' + ('m' if key == 'mesh_size_m' else ''))

    if re.search(r'辐射|radiation', prompt, re.I):
        cfg['radiation_enabled'] = True
        changes.append('启用表面辐射')
    if re.search(r'对流|换热|散热', prompt, re.I):
        cfg['heat_convection'] = True
        changes.append('热源表面启用默认对流散热')

    faces, face_note = _face_selection(model_id, prompt, int(model['triangles']))
    if face_note:
        if cfg.get('heat_sources'):
            source=cfg['heat_sources'][0]
            if source.get('source_type','surface')=='point':
                source['faces']=[]
                source['position_m']=_position_from_prompt(model_id,prompt,faces)
            else:
                source['faces'] = faces
        changes.append(face_note)
    if cfg.get('heat_sources'):
        source=cfg['heat_sources'][0]
        source.setdefault('source_type',source_type);source.setdefault('placement',placement)
        if source.get('source_type')=='point' or source.get('placement')=='embedded':
            source['position_m']=source.get('position_m') or _position_from_prompt(model_id,prompt,faces)
            source.setdefault('radius_m',max(model['dimensions_m'])/30)
    minimum_mesh=max(model['dimensions_m'])/130
    recommended_mesh=max(max(model['dimensions_m'])/30,minimum_mesh)
    if cfg.get('mesh_size_m',.035)<recommended_mesh:
        cfg['mesh_size_m']=recommended_mesh
        warnings.append(f'目标网格对当前模型过细，已按可运行规模调整为 {recommended_mesh*1000:.2f} mm；如需更细网格请缩短时长或增大保存间隔。')
    if cfg.get('heat_sources') and cfg['heat_sources'][0].get('source_type','surface')=='surface' and cfg['heat_sources'][0].get('placement','surface')!='embedded' and not cfg['heat_sources'][0].get('faces'):
        questions.append('请指定受热面：可在三维视图刷选，或在描述中写“整个外表面”“左侧/顶部”或“x=10 mm”。')
    if not cfg.get('heat_sources') and power is None:
        warnings.append('没有识别到热源功率；当前算例可能保持等温。')
    if cfg.get('duration_s', 3600) / max(cfg.get('save_s', 15), 1) > 300:
        warnings.append('保存帧数超过 301，已建议把保存间隔调大。')
    try:
        validated = Simulation.model_validate(cfg)
    except Exception as error:
        questions.append('参数组合需要调整：' + str(error).split('\n')[0])
        return dict(ok=False, config=cfg, changes=changes, warnings=warnings, questions=questions)
    return dict(ok=not questions, config=validated.model_dump(mode='json'), changes=changes, warnings=warnings, questions=questions)


def plan_request(model_id, prompt, current):
    if not prompt or len(prompt.strip()) < 2:
        return dict(ok=False, config=current, changes=[], warnings=[], questions=['请描述材料、热源功率、时长和受热面。'])
    return make_plan(model_id, prompt.strip(), current)
