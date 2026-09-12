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


def _codex_context(model_id, prompt, current):
    metadata = read_json(MODELS / model_id / 'metadata.json')
    geometry = {
        'name': metadata.get('name'),
        'kind': metadata.get('kind'),
        'dimensions_m': metadata.get('dimensions_m'),
        'triangles': metadata.get('triangles'),
        'components': metadata.get('components', []),
    }
    schema = Simulation.model_json_schema()
    return schema, geometry


def _codex_prompt(model_id, prompt, current):
    schema, geometry = _codex_context(model_id, prompt, current)
    instructions = (
        '你是 Thermal Studio 的仿真配置助手。根据用户描述生成完整、可执行的 Simulation JSON 配置。'
        '只输出一个 JSON 对象，不要 Markdown、解释或额外字段。必须符合给定 JSON Schema；'
        '保留当前配置中未被用户修改的字段。不要虚构不存在的表面编号；无法确定时在 JSON 中保留原值。'
    )
    user_input = {
        'model_id': model_id,
        'geometry': geometry,
        'current_config': current or {},
        'request': prompt.strip(),
        'simulation_schema': schema,
    }
    return instructions + '\n输入数据：\n' + json.dumps(user_input, ensure_ascii=False)


def _validated_codex_text(text, current):
    if not text:
        raise ValueError('Codex 未返回文本结果')
    config = _json_from_text(text)
    validated = Simulation.model_validate(config)
    return validated.model_dump(mode='json')


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
        if isinstance(item, dict) and item.get('type') == 'agent_message' and isinstance(item.get('text'), str):
            chunks.append(item['text'])
    return '\n'.join(chunks).strip()


def _codex_cli_plan_request(model_id, prompt, current, executable):
    try:
        cli_prompt = _codex_prompt(model_id, prompt, current)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'无法读取模型元数据：{error}'], mode='codex', provider='cli')
    args = [executable, 'exec', '--ephemeral', '--skip-git-repo-check', '--json', '--color', 'never', '-']
    # Let the official CLI use the model/profile selected in ~/.codex unless
    # the service explicitly overrides it for this bridge.
    model = os.environ.get('THERMAL_CODEX_MODEL', '').strip()
    if model:
        args[2:2] = ['--model', model]
    try:
        completed = subprocess.run(
            args, input=cli_prompt, text=True, encoding='utf-8', capture_output=True,
            cwd=str(ROOT), timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'无法启动本机 Codex CLI：{error}'], mode='codex', provider='cli')
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or '').strip()[-800:]
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'本机 Codex CLI 请求失败（退出码 {completed.returncode}）：{detail}'],
                    mode='codex', provider='cli')
    try:
        config = _validated_codex_text(_cli_response_text(completed.stdout), current)
    except Exception as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'Codex 返回的配置无法通过 Simulation 校验：{str(error).split(chr(10))[0]}'],
                    mode='codex', provider='cli')
    return dict(ok=True, config=config, changes=['已由本机 Codex 生成配置'],
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
        '保留当前配置中未被用户修改的字段。不要虚构不存在的表面编号；无法确定时在 JSON 中保留原值。'
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
        config = _validated_codex_text(text, current)
    except Exception as error:
        return dict(ok=False, config=current or {}, changes=[], warnings=[],
                    questions=[f'Codex 返回的配置无法通过 Simulation 校验：{str(error).split(chr(10))[0]}'], mode='codex', provider='api')
    return dict(ok=True, config=config, changes=['已由 Codex 生成配置'],
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
