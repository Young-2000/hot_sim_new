import copy
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from runtime import MODELS, ROOT, read_json
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
    # A heat source phrased as "在组件 9 施加" should target that
    # component's exterior faces, even when the user does not also say
    # "整个外表面".  Material assignments may mention several component
    # numbers, so only use a component preceded by an explicit target word.
    target = re.search(r'(?:在|给|对)\s*(?:组件|部件)\s*(\d+)', text)
    if not target:
        target = re.search(r'(?:组件|部件)\s*(\d+)\s*(?:的)?\s*(?:外表面|表面)', text)
    if target:
        component_id = int(target.group(1)) - 1
        metadata = read_json(MODELS / model_id / 'metadata.json')
        components = metadata.get('components') or []
        record = next((item for item in components if int(item.get('component_id', -1)) == component_id), None)
        if record:
            display = read_json(MODELS / model_id / 'display.json')
            points = np.asarray(display['points'], dtype=float).reshape(-1, 3)
            faces = np.asarray(display['faces'], dtype=np.int64).reshape(-1, 3)
            centers = points[faces].mean(axis=1)
            low = np.asarray(record['bounds_m'][0], dtype=float)
            high = np.asarray(record['bounds_m'][1], dtype=float)
            tolerance = max(float(np.ptp(points, axis=0).max()) * 1e-8, 1e-9)
            selected = np.all((centers >= low - tolerance) & (centers <= high + tolerance), axis=1)
            outer = np.asarray(display.get('is_outer', np.ones(len(centers))), dtype=bool)
            if len(outer) == len(selected):
                selected &= outer
            indices = np.flatnonzero(selected).tolist()
            if indices:
                return indices, f'已自动选择组件 {component_id + 1} 外表面 ({len(indices)} 个三角面)'
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

    component_aliases=(
        ('不锈钢|stainless','不锈钢（示例）'),('碳钢|steel','碳钢（示例）'),
        ('铝|alum|aluminium','铝（示例）'),('铜|copper','铜 C11000'),('铁|iron','铁（示例）'))
    assignments=[]
    for match in re.finditer(r'(?:组件|部件)\s*(\d+)[^，,。;；\n]{0,24}?(不锈钢|stainless|碳钢|steel|铝|alum|aluminium|铜|copper|铁|iron)',prompt,re.I):
        component_id=int(match.group(1))-1
        if component_id<0: continue
        material_name=next((name for pattern,name in component_aliases if re.fullmatch(pattern,match.group(2),re.I)),None)
        if material_name:
            preset=next(item for item in PRESETS if item['name']==material_name)
            assignments.append(dict(component_id=component_id,material=copy.deepcopy(preset)))
            changes.append(f'组件 {component_id+1} 材料: {material_name}')
    if assignments: cfg['component_materials']=assignments

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
        questions.append('草案缺少热源：请明确热源功率与受热面，或点热源位置，再生成配置。')
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


def _response_text(payload):
    """Extract text from the Responses API's output content blocks."""
    direct = payload.get('output_text')
    if isinstance(direct, str) and direct.strip():
        return direct
    chunks = []
    for item in payload.get('output') or []:
        for content in item.get('content') or []:
            if content.get('type') in ('output_text', 'text') and isinstance(content.get('text'), str):
                chunks.append(content['text'])
    return '\n'.join(chunks).strip()


def _json_from_text(text):
    """Parse JSON while tolerating a fenced block from a model."""
    value = text.strip()
    if value.startswith('```'):
        value = re.sub(r'^```(?:json)?\s*', '', value, flags=re.I)
        value = re.sub(r'\s*```$', '', value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find('{'), value.rfind('}')
        if start >= 0 and end > start:
            return json.loads(value[start:end + 1])
        raise


def _surface_selections(model_id):
    """Resolve named exterior patches to actual imported display-face IDs.

    Axis patches use exterior triangle centroids in the outermost 3% of that
    axis, with normals facing the requested direction. No guessed face IDs or
    fallback to the entire surface is used for an unavailable patch.
    """
    display = read_json(MODELS / model_id / 'display.json')
    points = np.asarray(display['points'], dtype=float).reshape(-1, 3)
    faces = np.asarray(display['faces'], dtype=np.int64).reshape(-1, 3)
    tri = points[faces]
    centers = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(lengths[:, None], 1e-30)
    outer = np.asarray(display.get('is_outer', np.ones(len(faces))), dtype=bool)
    valid = outer & (lengths > 0)
    selections = {}

    def add(key, name, mask):
        ids = np.flatnonzero(mask).tolist()
        selections[key] = dict(name=name, faces=ids, face_count=len(ids),
                               area_m2=float(lengths[mask].sum() / 2))

    add('all_outer', '全部外表面', valid)
    for key, name, axis, sign in (
        ('top', '顶部 (+Z)', 2, 1), ('bottom', '底部 (-Z)', 2, -1),
        ('right', '右侧 (+X)', 0, 1), ('left', '左侧 (-X)', 0, -1),
        ('back', '后侧 (+Y)', 1, 1), ('front', '前侧 (-Y)', 1, -1),
    ):
        mask = np.zeros(len(faces), dtype=bool)
        if np.any(valid):
            values = centers[valid, axis]
            edge = values.max() if sign > 0 else values.min()
            tolerance = max(float(np.ptp(values)) * .03, 1e-10)
            mask = valid & (np.abs(centers[:, axis] - edge) <= tolerance) & (normals[:, axis] * sign > .35)
        add(key, name, mask)
    return selections


def _codex_context(model_id, prompt, current):
    metadata = read_json(MODELS / model_id / 'metadata.json')
    geometry = {
        'name': metadata.get('name'),
        'kind': metadata.get('kind'),
        'dimensions_m': metadata.get('dimensions_m'),
        'triangles': metadata.get('triangles'),
        'components': metadata.get('components', []),
        'surface_selections': [dict(id=key, **{k: v for k, v in selection.items() if k != 'faces'})
                               for key, selection in _surface_selections(model_id).items()],
        'selection_rule': '轴向选区取最外侧 3% 范围内、法向朝向该方向的外表面三角面；编号由后端映射。',
    }
    schema = Simulation.model_json_schema()
    # The assistant uses semantic selectors; the solver still receives only
    # ordinary Simulation fields with resolved triangle IDs.
    selector = dict(type='string', enum=['top', 'bottom', 'right', 'left', 'back', 'front', 'all_outer'])
    for name in ('Heat', 'Cooling'):
        schema['$defs'][name]['properties']['surface_selection'] = selector
    schema['$defs']['Cooling']['required'] = [key for key in schema['$defs']['Cooling']['required'] if key != 'faces']
    schema['properties']['heat_sources']['minItems'] = 1
    schema['required'] = list(dict.fromkeys([*schema.get('required', []), 'heat_sources']))
    return schema, geometry


def _codex_prompt(model_id, prompt, current):
    schema, geometry = _codex_context(model_id, prompt, current)
    instructions = (
        '你是 Thermal Studio 的仿真配置助手。根据用户描述生成完整、可执行的 Simulation JSON 配置。'
        '只输出一个符合给定 JSON Schema 的 JSON 对象，不要 Markdown 或解释。不要调用工具、读取文件或运行仿真。'
        '保留当前配置中未被用户修改的字段和 model_id。必须包含至少一个有效热源，不能用空热源列表代替用户要求。'
        '新增或修改面选区时，使用 geometry.surface_selections 中的 id，写入热源或散热区的 surface_selection 字段；'
        '例如顶部热源使用 surface_selection="top"，不需要提供 faces。只可使用 face_count 大于零的选区。'
        '已有选区可保留原 faces，不要猜测或生成新的数字编号。热源位置与散热位置分别处理。'
        '全部外表面对流通常用 default_h 和 heat_convection；不得因此把顶部热源扩大为全部外表面。'
        '材料优先采用提供的材料预设。用户确认操作在网页执行，你只返回配置。'
    )
    user_input = {
        'model_id': model_id,
        'geometry': geometry,
        'current_config': current or {},
        'request': prompt.strip(),
        'material_presets': PRESETS,
        'simulation_schema': schema,
    }
    return instructions + '\n输入数据：\n' + json.dumps(user_input, ensure_ascii=False)


def _validated_codex_text(text, current, model_id=None):
    if not text:
        raise ValueError('Codex 未返回文本结果')
    returned = _json_from_text(text)
    if not isinstance(returned, dict):
        raise ValueError('配置必须是 JSON 对象。')
    config = {**copy.deepcopy(current or {}), **returned}
    expected_model = model_id or (current or {}).get('model_id')
    if expected_model and config.get('model_id', expected_model) != expected_model:
        raise ValueError('Agent 返回了其他模型的配置，请为当前模型重新生成。')
    if expected_model:
        config['model_id'] = expected_model
    if not config.get('heat_sources'):
        raise ValueError('草案未生成热源。请明确功率与受热面，或指定点热源位置；当前草案不能运行。')
    selections = None
    for group in [*config.get('heat_sources', []), *config.get('cooling', [])]:
        selector = group.pop('surface_selection', None)
        if selector is None:
            continue
        if selections is None:
            selections = _surface_selections(config['model_id'])
        if selector not in selections or not selections[selector]['faces']:
            raise ValueError(f'当前模型没有可用的“{selector}”受热/散热选区，请手动刷选后重新生成。')
        selected = selections[selector]['faces']
        if group.get('faces') and sorted(set(group['faces'])) != selected:
            raise ValueError('Agent 的选区名称与面编号冲突，请重新生成或手动刷选。')
        group['faces'] = selected
    validated = Simulation.model_validate(config)
    if model_id:
        metadata = read_json(MODELS / model_id / 'metadata.json')
        for group in [*validated.heat_sources, *validated.cooling]:
            if group.faces and max(group.faces) >= metadata['triangles']:
                raise ValueError('Agent 返回了当前模型不存在的面编号，请重新选取。')
    return validated.model_dump(mode='json')


def _codex_changes(config):
    changes = [f'材料：{config["base_material"]["name"]}',
               f'仿真时长：{config["duration_s"]:g} s；计算步长：{config["dt_s"]:g} s；保存间隔：{config["save_s"]:g} s']
    for heat in config['heat_sources']:
        target = (f'{len(heat["faces"])} 个三角面' if heat['source_type'] == 'surface' and heat['placement'] != 'embedded'
                  else f'点位置 {heat.get("position_m")} m')
        changes.append(f'热源“{heat["name"]}”：{heat["power_W"]:g} W，{heat["start_s"]:g}–{heat["end_s"]:g} s，{target}')
    return changes


def _find_codex_cli():
    configured = os.environ.get('THERMAL_CODEX_COMMAND', '').strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        found = shutil.which(configured)
        if found:
            return found
    for name in ('codex.exe', 'codex'):
        found = shutil.which(name)
        if found:
            return found
    local_app_data = os.environ.get('LOCALAPPDATA', '')
    if local_app_data:
        candidates = sorted(Path(local_app_data).glob('Programs/OpenAI Codex CLI/*/bin/codex.exe'), reverse=True)
        if candidates:
            return str(candidates[0])
    return None


def _cli_response_text(stdout):
    """Extract the final agent message from Codex CLI JSONL events."""
    chunks = []
    for line in (stdout or '').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get('item') if isinstance(event, dict) else None
        if isinstance(event, dict) and event.get('type') == 'item.completed' and isinstance(item, dict) and item.get('type') == 'agent_message' and isinstance(item.get('text'), str):
            chunks.append(item['text'])
    return chunks[-1].strip() if chunks else ''


def _codex_cli_env():
    """Give this child the user's proxy settings without changing the parent.

    The Windows browser uses WinINET proxy settings, while the CLI needs proxy
    environment variables. Explicit proxy environment settings take precedence.
    Read the registry for each request so a running service sees proxy changes.
    """
    env = os.environ.copy()
    explicit = {key.lower() for key in env if key.lower() in
                ('http_proxy', 'https_proxy', 'all_proxy')}
    registry_proxies = getattr(urllib.request, 'getproxies_registry', lambda: {})
    if not explicit:
        for scheme, address in registry_proxies().items():
            if scheme in ('http', 'https') and address:
                env[f'{scheme.upper()}_PROXY'] = address
    exclusions = env.get('no_proxy', env.get('NO_PROXY', '')).split(',')
    exclusions = [entry.strip() for entry in exclusions if entry.strip()]
    for host in ('127.0.0.1', 'localhost', '::1'):
        if host not in exclusions:
            exclusions.append(host)
    env['NO_PROXY'] = env['no_proxy'] = ','.join(exclusions)
    return env


def _codex_cli_timeout():
    raw = os.environ.get('THERMAL_CODEX_TIMEOUT_S', '180')
    try:
        seconds = int(raw)
    except ValueError as error:
        raise ValueError('THERMAL_CODEX_TIMEOUT_S 必须是 15–600 之间的整数秒数。') from error
    if not 15 <= seconds <= 600:
        raise ValueError('THERMAL_CODEX_TIMEOUT_S 必须是 15–600 之间的整数秒数。')
    return seconds


def _cli_diagnostic_text(value):
    # TimeoutExpired may contain bytes even when subprocess uses text=True.
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    text = value or ''
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    text = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [redacted]', text)
    return re.sub(r'\bsk-[A-Za-z0-9_-]+', '[redacted]', text)


def _cli_error_detail(stdout, stderr):
    """Prefer structured CLI errors over startup warnings on stderr."""
    stdout = _cli_diagnostic_text(stdout)
    stderr = _cli_diagnostic_text(stderr)
    for line in reversed((stdout or '').splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            if event.get('type') == 'error' and isinstance(event.get('message'), str):
                return event['message']
            error = event.get('error')
            if isinstance(error, dict) and isinstance(error.get('message'), str):
                return error['message']
    for line in reversed(stderr.splitlines()):
        if any(term in line.lower() for term in ('request timed out', 'stream disconnected', 'connection refused', 'error:')):
            return line.strip()[-500:]
    return (stderr or stdout or '').strip()[-800:]


def _codex_cli_plan_request(model_id, prompt, current, executable):
    try:
        cli_prompt = _codex_prompt(model_id, prompt, current)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'无法读取模型元数据：{error}'], mode='codex', provider='cli')
    args = [executable, 'exec', '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only', '--json', '--color', 'never', '-']
    # Let the official CLI use the model/profile selected in ~/.codex unless
    # the service explicitly overrides it for this bridge.
    model = os.environ.get('THERMAL_CODEX_MODEL', '').strip()
    if model:
        args[2:2] = ['--model', model]
    try:
        timeout = _codex_cli_timeout()
    except ValueError as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[str(error)], mode='codex', provider='cli', error_code='configuration')
    try:
        completed = subprocess.run(
            args, input=cli_prompt, text=True, encoding='utf-8', capture_output=True,
            cwd=str(ROOT), timeout=timeout, env=_codex_cli_env(),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
    except subprocess.TimeoutExpired as error:
        detail = _cli_error_detail(error.stdout, error.stderr)
        network_timeout = any(term in detail.lower() for term in ('request timed out', 'stream disconnected'))
        message = (f'Codex 上游连接超时，重试后仍未在 {timeout} 秒内完成。请检查网络或系统代理。'
                   if network_timeout else f'Codex 配置生成超过 {timeout} 秒，尚未收到完整结果。请稍后重试。')
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[message], mode='codex', provider='cli', error_code='timeout', diagnostic=detail)
    except OSError as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'无法启动本机 Codex CLI：{error}'], mode='codex', provider='cli', error_code='startup')
    if completed.returncode != 0:
        detail = _cli_error_detail(completed.stdout, completed.stderr)
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'本机 Codex CLI 请求失败（退出码 {completed.returncode}）：{detail}'],
                    mode='codex', provider='cli', error_code='cli_failed')
    try:
        config = _validated_codex_text(_cli_response_text(completed.stdout), current, model_id)
    except Exception as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'Codex 返回的配置无法通过 Simulation 校验：{str(error).split(chr(10))[0]}'],
                    mode='codex', provider='cli', error_code='invalid_config')
    return dict(ok=True, config=config, changes=_codex_changes(config),
                warnings=[], questions=[], mode='codex', provider='cli', model=model or None)


def _codex_api_plan_request(model_id, prompt, current):
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=['Codex API 模式需要设置 OPENAI_API_KEY；当前未配置。'],
                    mode='codex', provider='api')
    model = os.environ.get('OPENAI_MODEL', 'gpt-5.2').strip() or 'gpt-5.2'
    base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').strip().rstrip('/')
    try:
        schema, geometry = _codex_context(model_id, prompt, current)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'无法读取模型元数据：{error}'], mode='codex', provider='api')
    instructions = (
        '你是 Thermal Studio 的仿真配置助手。根据用户描述生成完整、可执行的 Simulation JSON 配置。'
        '只输出一个 JSON 对象，不要 Markdown、解释或额外字段。必须符合给定 JSON Schema；'
        '保留当前配置中未被用户修改的字段与 model_id。必须生成至少一个有效热源。'
        '新增或修改受热面时使用 geometry.surface_selections 中非空选区的 id，写入 surface_selection 字段，'
        '例如顶部热源使用 surface_selection="top"，由本地后端生成 faces；不要猜测面编号。'
        '热源与散热面分别处理，全部外表面对流使用 default_h 与 heat_convection。用户确认由网页处理，只返回配置。'
    )
    user_input = {
        'model_id': model_id,
        'geometry': geometry,
        'current_config': current or {},
        'request': prompt.strip(),
        'simulation_schema': schema,
    }
    body = json.dumps({
        'model': model,
        'input': [
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': json.dumps(user_input, ensure_ascii=False)},
        ],
        'text': {
            'format': {
                'type': 'json_schema',
                'name': 'simulation_config',
                # Pydantic defaults are optional in its generated schema;
                # non-strict structured output keeps those defaults valid,
                # while Simulation.model_validate below remains authoritative.
                'strict': False,
                'schema': schema,
            }
        },
    }, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(
        f'{base_url}/responses', data=body,
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')[:500]
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'Codex 请求失败（HTTP {error.code}）：{detail}'], mode='codex', provider='api')
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'无法连接 Codex API：{error}'], mode='codex', provider='api')
    try:
        text = _response_text(payload)
        if not text:
            raise ValueError('Responses API 未返回文本结果')
        config = _validated_codex_text(text, current, model_id)
    except Exception as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'Codex 返回的配置无法通过 Simulation 校验：{str(error).split(chr(10))[0]}'], mode='codex', provider='api')
    return dict(ok=True, config=config, changes=_codex_changes(config),
                warnings=[], questions=[], mode='codex', provider='api', model=model)


def codex_plan_request(model_id, prompt, current):
    """Generate a validated Simulation config using the local Codex client by default.

    Set THERMAL_CODEX_PROVIDER=api to use the legacy Responses API. ``auto``
    uses the local client when available and otherwise requires an API key.
    """
    provider = os.environ.get('THERMAL_CODEX_PROVIDER', 'cli').strip().lower() or 'cli'
    if provider not in ('cli', 'api', 'auto'):
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=['THERMAL_CODEX_PROVIDER 必须是 cli、api 或 auto。'], mode='codex')
    if provider in ('cli', 'auto'):
        executable = _find_codex_cli()
        if executable:
            return _codex_cli_plan_request(model_id, prompt, current, executable)
        if provider == 'cli':
            return dict(ok=False, config=current or {}, changes=[], warnings=[],
                        questions=['未找到本机 Codex CLI（codex.exe）。请安装官方 Codex CLI，或设置 THERMAL_CODEX_PROVIDER=api。'],
                        mode='codex', provider='cli')
    return _codex_api_plan_request(model_id, prompt, current)
