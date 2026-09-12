from runtime import *
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Annotated, Literal
Finite = Annotated[float, Field(allow_inf_nan=False)]

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Material(Strict):
    name: str = Field(min_length=1, max_length=60)
    k: Finite = Field(gt=0, le=10000)
    rho: Finite = Field(gt=0, le=100000)
    cp: Finite = Field(gt=0, le=100000)
    # Isotropic linear thermal expansion coefficient, 1/K.  Zero keeps
    # backwards compatibility for materials without expansion data.
    thermal_expansion_CTE_per_K: Finite = Field(default=0, ge=0, le=1)
    phase_change: 'PhaseChange | None' = None

class PhaseChange(Strict):
    melting_C: Finite = Field(ge=-273.15, le=5000)
    mushy_C: Finite = Field(gt=0, le=1000)
    latent_J_kg: Finite = Field(gt=0, le=1e9)

    @model_validator(mode='after')
    def valid(self):
        if self.melting_C + self.mushy_C > 5000: raise ValueError('相变温区超出允许范围')
        return self

PRESETS = [
    dict(name='铜 C11000',k=391,rho=8910,cp=385,thermal_expansion_CTE_per_K=16.5e-6),
    dict(name='铝（示例）',k=205,rho=2700,cp=900,thermal_expansion_CTE_per_K=23.1e-6),
    dict(name='铁（示例）',k=80,rho=7870,cp=449,thermal_expansion_CTE_per_K=11.8e-6),
    dict(name='碳钢（示例）',k=50,rho=7850,cp=470,thermal_expansion_CTE_per_K=12e-6),
    dict(name='不锈钢（示例）',k=16,rho=8000,cp=500,thermal_expansion_CTE_per_K=17.3e-6),
]

class Region(Strict):
    name: str = Field(default='材料区域',max_length=60)
    min_m: list[Finite] = Field(min_length=3,max_length=3)
    max_m: list[Finite] = Field(min_length=3,max_length=3)
    material: Material
    @model_validator(mode='after')
    def ordered(self):
        if any(a>=b for a,b in zip(self.min_m,self.max_m)): raise ValueError('区域每个轴的最小值必须小于最大值')
        return self

class ComponentMaterial(Strict):
    component_id: int = Field(ge=0, le=10000)
    material: Material

class Heat(Strict):
    name: str = Field(default='热源',max_length=60)
    source_type: Literal['point','surface'] = 'surface'
    placement: Literal['embedded','surface','external'] = 'surface'
    power_W: Finite = Field(gt=0,le=1e8)
    start_s: Finite = Field(default=0,ge=0)
    end_s: Finite = Field(default=3600,gt=0)
    faces: list[int] = Field(default_factory=list,max_length=500000)
    position_m: list[Finite] | None = Field(default=None,min_length=3,max_length=3)
    radius_m: Finite = Field(default=.001,gt=0,le=100)
    power_profile: list['PowerPoint'] = Field(default_factory=list,max_length=128)
    thermostat: 'Thermostat | None' = None
    @model_validator(mode='after')
    def ordered(self):
        if self.end_s<=self.start_s: raise ValueError('热源结束时间必须晚于开始时间')
        if self.source_type == 'surface' and self.placement != 'embedded' and not self.faces:
            raise ValueError('面热源必须选择至少一个表面三角面')
        if (self.source_type == 'point' or self.placement == 'embedded') and self.position_m is None and not self.faces:
            raise ValueError('点热源或嵌入热源必须指定位置')
        if self.faces and min(self.faces)<0: raise ValueError('无效的面编号')
        self.faces=sorted(set(self.faces))
        self.power_profile=sorted(self.power_profile,key=lambda p:p.time_s)
        return self

class PowerPoint(Strict):
    time_s: Finite = Field(ge=0, le=864000)
    power_W: Finite = Field(ge=0, le=1e8)

class Thermostat(Strict):
    target_C: Finite = Field(ge=-273.15, le=5000)
    hysteresis_C: Finite = Field(gt=0, le=1000)
    min_power_W: Finite = Field(ge=0, le=1e8, default=0)
    max_power_W: Finite = Field(gt=0, le=1e8)

class Cooling(Strict):
    name: str = Field(default='散热区',max_length=60)
    h: Finite = Field(ge=0,le=1e6)
    ambient_C: Finite = Field(ge=-273.15,le=5000)
    faces: list[int] = Field(min_length=1,max_length=500000)
    radiation: bool = False
    emissivity: Finite = Field(default=.85,ge=0,le=1)
    @model_validator(mode='after')
    def valid_faces(self):
        if min(self.faces)<0: raise ValueError('无效的面编号')
        self.faces=sorted(set(self.faces)); return self

class Simulation(Strict):
    model_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,60}$')
    name: str = Field(default='未命名算例',min_length=1,max_length=80)
    base_material: Material = Field(default_factory=lambda:Material(**PRESETS[0]))
    regions: list[Region] = Field(default_factory=list,max_length=16)
    component_materials: list[ComponentMaterial] = Field(default_factory=list,max_length=64)
    heat_sources: list[Heat] = Field(default_factory=list,max_length=16)
    cooling: list[Cooling] = Field(default_factory=list,max_length=16)
    analysis_mode: Literal['transient','steady'] = 'transient'
    initial_C: Finite = Field(default=25,ge=-273.15,le=5000)
    ambient_C: Finite = Field(default=25,ge=-273.15,le=5000)
    radiation_ambient_C: Finite = Field(default=25,ge=-273.15,le=5000)
    default_h: Finite = Field(default=10,ge=0,le=1e6)
    heat_convection: bool = False
    radiation_enabled: bool = False
    emissivity: Finite = Field(default=.8,ge=0,le=1)
    # Heat transfer through the air between disconnected solid components.
    # This is a reduced-order conduction model, not a CFD airflow solve.
    air_gap_enabled: bool = True
    air_gap_k_W_mK: Finite = Field(default=.026,gt=0,le=10)
    air_gap_max_m: Finite = Field(default=.05,gt=1e-7,le=10)
    contact_resistance_m2K_W: Finite = Field(default=0,ge=0,le=1e6)
    duration_s: Finite = Field(default=3600,gt=0,le=864000)
    dt_s: Finite = Field(default=15,gt=0,le=3600)
    save_s: Finite = Field(default=15,gt=0,le=864000)
    mesh_size_m: Finite = Field(default=.035,gt=1e-7,le=100)
    @model_validator(mode='after')
    def resources(self):
        if self.duration_s/self.dt_s>5000: raise ValueError('单次最多 5000 个时间步，请增大计算步长')
        if self.duration_s/self.save_s>300: raise ValueError('单次最多保存 301 帧，请增大保存间隔')
        return self

class AgentRequest(Strict):
    model_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,60}$')
    prompt: str = Field(min_length=2, max_length=4000)
    config: dict = Field(default_factory=dict)

class CalibrationPoint(Strict):
    time_s: Finite = Field(ge=0, le=864000)
    temperature_C: Finite = Field(ge=-273.15, le=5000)

class CalibrationRequest(Strict):
    mode: Literal['fit_h','power_limit']
    mass_kg: Finite = Field(gt=0, le=1e9)
    cp_J_kgK: Finite = Field(gt=0, le=1e7)
    area_m2: Finite = Field(gt=0, le=1e9)
    initial_C: Finite = Field(default=25, ge=-273.15, le=5000)
    ambient_C: Finite = Field(default=25, ge=-273.15, le=5000)
    power_W: Finite = Field(default=100, ge=0, le=1e12)
    h_W_m2K: Finite = Field(default=10, ge=0, le=1e6)
    max_temperature_C: Finite = Field(default=100, ge=-273.15, le=5000)
    duration_s: Finite = Field(default=3600, gt=0, le=864000)
    h_bounds: list[Finite] = Field(default_factory=lambda:[.01,10000], min_length=2, max_length=2)
    measurements: list[CalibrationPoint] = Field(default_factory=list, min_length=2, max_length=10000)

class RefrigerationCycle(Strict):
    refrigerant: Literal['R134a','R410A','R32','R290','ideal'] = 'ideal'
    evaporating_C: Finite = Field(default=5, gt=-100, lt=100)
    condensing_C: Finite = Field(default=45, gt=-50, lt=150)
    superheat_C: Finite = Field(default=5, ge=0, le=100)
    subcool_C: Finite = Field(default=3, ge=0, le=100)
    cooling_capacity_W: Finite = Field(default=1000, gt=0, le=1e9)
    compressor_efficiency: Finite = Field(default=.75, gt=0, le=1)
    fan_pump_power_W: Finite = Field(default=0, ge=0, le=1e9)

    @model_validator(mode='after')
    def valid_temperatures(self):
        if self.condensing_C <= self.evaporating_C + 1: raise ValueError('冷凝温度必须高于蒸发温度')
        return self

class CpuSimulation(Strict):
    """Standalone reduced-order CPU cooling scenario.

    This intentionally complements the mesh solver: it represents a motherboard,
    CPU package, cooler and (for water cooling) coolant/radiator as thermal nodes.
    """
    cooler_type: Literal['air', 'water'] = 'air'
    duration_s: Finite = Field(default=600, gt=0, le=864000)
    dt_s: Finite = Field(default=0.5, gt=0, le=60)
    initial_C: Finite = Field(default=25, ge=-273.15, le=5000)
    ambient_C: Finite = Field(default=25, ge=-273.15, le=5000)
    cpu_power_W: Finite = Field(default=125, ge=0, le=1e5)
    cpu_mass_kg: Finite = Field(default=.05, gt=0, le=100)
    cpu_cp_J_kgK: Finite = Field(default=700, gt=0, le=1e6)
    board_mass_kg: Finite = Field(default=.6, gt=0, le=100)
    board_cp_J_kgK: Finite = Field(default=900, gt=0, le=1e6)
    interface_resistance_K_W: Finite = Field(default=.08, gt=0, le=100)
    board_h_W_m2K: Finite = Field(default=8, ge=0, le=1e6)
    board_area_m2: Finite = Field(default=.08, gt=0, le=100)
    cooler_mass_kg: Finite = Field(default=.25, gt=0, le=100)
    cooler_cp_J_kgK: Finite = Field(default=385, gt=0, le=1e6)
    air_h_W_m2K: Finite = Field(default=80, ge=0, le=1e6)
    air_area_m2: Finite = Field(default=.12, gt=0, le=100)
    air_capacity_W: Finite = Field(default=250, ge=0, le=1e6)
    water_flow_L_min: Finite = Field(default=2, gt=0, le=1e5)
    water_inlet_C: Finite = Field(default=25, ge=-273.15, le=5000)
    water_cp_J_kgK: Finite = Field(default=4182, gt=0, le=1e6)
    water_cooler_UA_W_K: Finite = Field(default=180, ge=0, le=1e7)
    radiator_UA_W_K: Finite = Field(default=120, ge=0, le=1e7)

    @model_validator(mode='after')
    def valid_steps(self):
        if self.duration_s / self.dt_s > 200000:
            raise ValueError('CPU 仿真最多 200000 个时间步，请增大步长或缩短时长')
        return self

Material.model_rebuild()
Heat.model_rebuild()
