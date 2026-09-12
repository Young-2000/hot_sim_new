from runtime import *
import csv, time, zipfile
import numpy as np
import trimesh, h5py
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from skfem import MeshTet, Basis, ElementTetP1, BilinearForm, asm
from skfem.helpers import dot, grad
from geometry import project, build_mesh

SIGMA = 5.670374419e-8

@BilinearForm
def capacity(u,v,w):return w.capacity*u*v
@BilinearForm
def conductivity(u,v,w):return w.conductivity*dot(grad(u),grad(v))

def _point_source_load(points, candidates, position, radius):
    """Spread a point input over nearby FEM nodes while preserving total power."""
    distances=np.linalg.norm(points[candidates]-np.asarray(position,dtype=float),axis=1)
    span=max(float(np.ptp(points,axis=0).max()),1e-9)
    spread=max(float(radius or 0),span/500)
    selected=distances<=3*spread
    if not np.any(selected): selected[np.argmin(distances)]=True
    weights=np.exp(-np.square(distances[selected]/spread)/2)
    weights/=weights.sum()
    load=np.zeros(len(points),dtype=float);load[np.asarray(candidates)[selected]]=weights
    return load

def assemble(mesh,cfg,display):
    points,tets=mesh['points'],mesh['tets'];centers=points[tets].mean(1)
    materials=[cfg['base_material']]+[r['material'] for r in cfg['regions']]
    labels=np.zeros(len(tets),dtype=np.int32)
    for i,r in enumerate(cfg['regions'],1):
        inside=np.all((centers>=np.asarray(r['min_m']))&(centers<=np.asarray(r['max_m'])),axis=1);labels[inside]=i
    vol=mesh['tet_volumes']
    k_values=np.array([m['k'] for m in materials],dtype=float)[labels]
    contact_r=float(cfg.get('contact_resistance_m2K_W',0) or 0)
    if contact_r and len(materials)>1:
        # The conforming mesh shares interface nodes. Smear an equivalent
        # interface layer over region cells to represent finite resistance.
        characteristic=np.maximum((6*vol/np.pi)**(1/3),1e-12)
        k_values[labels>0]=1/(1/k_values[labels>0]+contact_r/characteristic[labels>0])
    props=np.array([[m['rho']*m['cp'],m['k'],m['rho']] for m in materials])[labels]
    props[:,1]=k_values
    basis=Basis(MeshTet(points.T,tets.T),ElementTetP1())
    M=asm(capacity,basis,capacity=props[:,0,None]).tocsr()
    K=asm(conductivity,basis,conductivity=props[:,1,None]).tocsr()
    bf=mesh['boundary_triangles'];tri=points[bf]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    queries=np.einsum('qn,fnc->fqc',bary,tri).reshape(-1,3)
    original=trimesh.Trimesh(vertices=display['points'],faces=display['faces'],process=False)
    _,dist,ids=project(original,queries)
    ids=ids.reshape(-1,3);exterior=display['is_outer'][ids]
    loads=[];heat_audits=[];is_heat=np.zeros(ids.shape,dtype=bool)
    display_tri=np.asarray(display['points'])[np.asarray(display['faces'])]
    display_centers=display_tri.mean(axis=1)
    boundary_nodes=np.unique(bf.ravel())
    for heat in cfg['heat_sources']:
        source_type=heat.get('source_type','surface');placement=heat.get('placement','surface')
        if source_type=='point' or placement=='embedded':
            if heat.get('position_m') is not None: position=np.asarray(heat['position_m'],dtype=float)
            elif heat.get('faces'): position=display_centers[np.asarray(heat['faces'],dtype=int)].mean(axis=0)
            else: raise ValueError(f'热源“{heat["name"]}”没有有效位置。')
            candidates=np.arange(len(points)) if placement=='embedded' else boundary_nodes
            F=_point_source_load(points,candidates,position,heat.get('radius_m',.001))*float(heat['power_W']);loads.append(F)
            selected=np.zeros(len(display['faces']),dtype=bool)
            selected[heat['faces']]=True
            if not heat.get('faces'):
                selected[int(np.argmin(np.linalg.norm(display_centers-position,axis=1)))]=True
            is_heat|=selected[ids]
            heat_audits.append(dict(name=heat['name'],source_type=source_type,placement=placement,power_W=float(F.sum()),position_m=position.tolist(),radius_m=float(heat.get('radius_m',.001))))
            continue
        selected=np.zeros(len(display['faces']),dtype=bool);selected[heat['faces']]=True
        hot=selected[ids];is_heat|=hot;hotarea=float((area[:,None]/3*hot).sum())
        if hotarea<=0:raise ValueError(f'热源“{heat["name"]}”的选区小于当前网格分辨率，请扩大选区或细化网格。')
        local=np.einsum('fq,qi->fi',area[:,None]/3*hot*heat['power_W']/hotarea,bary)
        F=np.bincount(bf.ravel(),weights=local.ravel(),minlength=len(points));loads.append(F)
        stltri=display['points'][display['faces'][heat['faces']]]
        originalarea=float(np.linalg.norm(np.cross(stltri[:,1]-stltri[:,0],stltri[:,2]-stltri[:,0]),axis=1).sum()/2)
        heat_audits.append(dict(name=heat['name'],source_type=source_type,placement=placement,power_W=float(F.sum()),selected_area_m2=originalarea,mapped_area_m2=hotarea))
    h=np.where(exterior,cfg['default_h'],0.).astype(float)
    ambient=np.full(ids.shape,cfg['ambient_C'])
    radiation=np.full(ids.shape,bool(cfg.get('radiation_enabled',False)),dtype=bool)
    radiation_ambient=np.full(ids.shape,float(cfg.get('radiation_ambient_C',cfg['ambient_C'])))
    emissivity=np.full(ids.shape,float(cfg.get('emissivity',.8)))
    if not cfg['heat_convection']:h[is_heat]=0
    for cool in cfg['cooling']:
        selected=np.zeros(len(display['faces']),dtype=bool);selected[cool['faces']]=True
        mask=selected[ids];h[mask]=cool['h'];ambient[mask]=cool['ambient_C']
        radiation[mask]=bool(cool.get('radiation',False));radiation_ambient[mask]=cool['ambient_C'];emissivity[mask]=cool.get('emissivity',.85)
    weights=area[:,None]/3*h
    local=np.einsum('fq,qi,qj->fij',weights,bary,bary)
    rows=np.broadcast_to(bf[:,:,None],local.shape);cols=np.broadcast_to(bf[:,None,:],local.shape)
    C=coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=M.shape).tocsr()
    ambientlocal=np.einsum('fq,qi->fi',weights*(ambient-cfg['initial_C']),bary)
    G=np.bincount(bf.ravel(),weights=ambientlocal.ravel(),minlength=len(points))
    rad_weights=area[:,None]/3*radiation
    rad_area=np.bincount(bf.ravel(),weights=rad_weights.ravel(),minlength=len(points))
    rad_ambient_sum=np.bincount(bf.ravel(),weights=(rad_weights*radiation_ambient).ravel(),minlength=len(points))
    rad_eps_sum=np.bincount(bf.ravel(),weights=(rad_weights*emissivity).ravel(),minlength=len(points))
    rad_ambient=np.divide(rad_ambient_sum,rad_area,out=np.full(len(points),cfg['ambient_C']),where=rad_area>0)
    rad_eps=np.divide(rad_eps_sum,rad_area,out=np.zeros(len(points)),where=rad_area>0)
    phase_entries=[]
    for i,m in enumerate(materials):
        phase=m.get('phase_change')
        if not phase: continue
        extra=np.zeros(len(points),dtype=float)
        cell_extra=vol[labels==i]*m['rho']*phase['latent_J_kg']/phase['mushy_C']
        np.add.at(extra,tets[labels==i].ravel(),np.repeat(cell_extra/4,4))
        phase_entries.append(dict(extra=extra,low=phase['melting_C'],high=phase['melting_C']+phase['mushy_C']))
    bmesh=trimesh.Trimesh(vertices=points,faces=bf,process=False)
    closest,vd,vi=project(bmesh,display['points'])
    vb=trimesh.triangles.points_to_barycentric(points[bf[vi]],closest)
    vol=mesh['tet_volumes']
    audit=dict(heat_sources=heat_audits,boundary_projection_max_m=float(dist.max()),display_projection_max_m=float(vd.max()),
        volume_m3=float(vol.sum()),mass_kg=float(vol@props[:,2]),heat_capacity_J_K=float(M.sum()),contact_resistance_m2K_W=contact_r,
        constant_field_conduction_residual=float(np.abs(K@np.ones(len(points))).max()),
        materials=[dict(**m,volume_m3=float(vol[labels==i].sum()),cells=int((labels==i).sum())) for i,m in enumerate(materials)])
    return M,K,C,G,loads,labels,bf[vi],vb,audit,rad_area,rad_ambient,rad_eps,phase_entries

def _power_at(source, t):
    profile=source.get('power_profile') or []
    if not profile: return float(source['power_W'])
    if t<=profile[0]['time_s']: return float(profile[0]['power_W'])
    for left,right in zip(profile[:-1],profile[1:]):
        if t<=right['time_s']:
            span=right['time_s']-left['time_s']
            return float(left['power_W']+(right['power_W']-left['power_W'])*(t-left['time_s'])/span) if span else float(right['power_W'])
    return float(profile[-1]['power_W'])


def _radiation_terms(theta, cfg, rad_area, rad_ambient, rad_eps):
    if rad_area is None or not np.any(rad_area): return None, None, np.zeros_like(theta)
    tk=np.maximum(theta+cfg['initial_C']+273.15,1e-6)
    ak=np.maximum(rad_ambient+273.15,1e-6)
    h=rad_eps*SIGMA*(tk+ak)*(tk*tk+ak*ak)
    coeff=rad_area*h
    return diags(coeff), coeff*(rad_ambient-cfg['initial_C']), coeff*(theta-(rad_ambient-cfg['initial_C']))


def _source_power(source, t0, t1, theta, cfg, state):
    if t1<=source['start_s'] or t0>=source['end_s']: return 0.
    mid=(t0+t1)/2
    power=_power_at(source,mid)
    control=source.get('thermostat')
    if control:
        average=float(theta.mean()+cfg['initial_C'])
        if state.get(id(source),True) and average>=control['target_C']:
            state[id(source)]=False
        elif not state.get(id(source),True) and average<=control['target_C']-control['hysteresis_C']:
            state[id(source)]=True
        power=control['max_power_W'] if state.get(id(source),True) else control.get('min_power_W',0.)
    return float(power)


def integrate(M,K,C,G,loads,cfg,callback=lambda *x:None,rad_area=None,rad_ambient=None,rad_eps=None,phase_entries=None):
    end=cfg['duration_s'];dt=cfg['dt_s'];save=cfg['save_s']
    output=np.unique(np.r_[np.arange(0,end,save),end])
    events=np.r_[0,end,np.arange(0,end,dt),output]
    for heat in cfg['heat_sources']:events=np.r_[events,heat['start_s'],heat['end_s']]
    grid=np.unique(np.round(events[(events>=0)&(events<=end)],10))
    theta=np.zeros(M.shape[0]);mc=np.asarray(M.sum(axis=0)).ravel();cc=np.asarray(C.sum(axis=0)).ravel()
    phase_entries=phase_entries or []
    saved=[];stats=[];integrated_in=0.;integrated_loss=0.;max_residual=0.;factor=None;last_key=None;control_state={}
    def stored_energy(values):
        energy=float(mc@values)
        for entry in phase_entries:
            energy+=float(np.sum(entry['extra']*np.clip(values+cfg['initial_C']-entry['low'],0,entry['high']-entry['low'])))
        return energy
    def snapshot(t):
        energy=stored_energy(theta);saved.append(theta.astype(np.float32))
        stats.append(dict(time_s=float(t),minimum_C=float(theta.min()+cfg['initial_C']),maximum_C=float(theta.max()+cfg['initial_C']),
            average_C=float(energy/mc.sum()+cfg['initial_C']),stored_energy_J=energy,convective_loss_W=float(cc@theta-G.sum())))
    snapshot(0);save_index=1
    for i,(t0,t1) in enumerate(zip(grid[:-1],grid[1:])):
        step=round(float(t1-t0),10)
        if step<=0:continue
        phase_extra=sum((entry['extra']*((theta+cfg['initial_C']>=entry['low'])&(theta+cfg['initial_C']<=entry['high'])) for entry in phase_entries),np.zeros_like(theta))
        Meff=M+diags(phase_extra) if np.any(phase_extra) else M
        R,Gr,rad_loss=_radiation_terms(theta,cfg,rad_area,rad_ambient,rad_eps)
        A=Meff/step+K+C+(R if R is not None else 0)
        key=(step, bool(np.any(phase_extra)), bool(R is not None))
        if R is not None or np.any(phase_extra) or factor is None or key!=last_key:
            factor=splu(A.tocsc());last_key=key
        F=G.copy();pin=0.
        for load,source in zip(loads,cfg['heat_sources']):
            power=_source_power(source,t0,t1,theta,cfg,control_state)
            if power and source['power_W']>0:
                F+=load*(power/source['power_W']);pin+=power
        before=stored_energy(theta);theta=factor.solve(Meff@theta/step+F+(Gr if Gr is not None else 0))
        if not np.isfinite(theta).all():raise ValueError('温度求解出现非有限值，请检查物性和网格。')
        rad_loss_new=float((np.asarray(R.diagonal())*(theta-(rad_ambient-cfg['initial_C']))).sum()) if R is not None else 0.
        loss=float(cc@theta-G.sum())+rad_loss_new
        integrated_in+=pin*step;integrated_loss+=loss*step
        stored=stored_energy(theta);max_residual=max(max_residual,abs(stored-before-step*(pin-loss)))
        if save_index<len(output) and abs(t1-output[save_index])<1e-7:snapshot(t1);save_index+=1
        if i%10==0:callback(float(t1/end),f'正在计算 {t1:.0f} / {end:.0f} s')
    balance=integrated_in-integrated_loss-stored_energy(theta)
    if abs(balance)>max(1.,abs(integrated_in)+abs(integrated_loss))*1e-6:raise ValueError('离散能量平衡检查未通过')
    return np.array(saved),output,stats,dict(input_energy_J=integrated_in,convective_energy_J=integrated_loss,energy_balance_error_J=balance,maximum_step_residual_J=max_residual)


def steady(M,K,C,G,loads,cfg,callback=lambda *x:None,rad_area=None,rad_ambient=None,rad_eps=None):
    theta=np.zeros(M.shape[0]);mc=np.asarray(M.sum(axis=0)).ravel();F=G.copy()
    for load,source in zip(loads,cfg['heat_sources']):
        power=_power_at(source,cfg['duration_s']) if source['start_s']<=cfg['duration_s']<=source['end_s'] else 0.
        if source['power_W']>0:F+=load*(power/source['power_W'])
    for iteration in range(80):
        R,Gr,_=_radiation_terms(theta,cfg,rad_area,rad_ambient,rad_eps)
        next_theta=splu((K+C+(R if R is not None else 0)).tocsc()).solve(F+(Gr if Gr is not None else 0))
        callback((iteration+1)/80,f'稳态迭代 {iteration+1} / 80')
        if np.max(np.abs(next_theta-theta))<1e-7:theta=next_theta;break
        theta=.65*theta+.35*next_theta
    final_R,final_Gr,_=_radiation_terms(theta,cfg,rad_area,rad_ambient,rad_eps)
    residual=float(np.max(np.abs((K+C+(final_R if final_R is not None else 0))@theta-F-(final_Gr if final_Gr is not None else 0))))
    stats=[dict(time_s=0.,minimum_C=float(cfg['initial_C']),maximum_C=float(cfg['initial_C']),average_C=float(cfg['initial_C']),stored_energy_J=0.,convective_loss_W=0.)]
    final=theta+cfg['initial_C'];stats.append(dict(time_s=float(cfg['duration_s']),minimum_C=float(final.min()),maximum_C=float(final.max()),average_C=float((mc@theta)/mc.sum()+cfg['initial_C']),stored_energy_J=float(mc@theta),convective_loss_W=float((C@theta-G).sum())))
    return np.asarray([np.zeros_like(theta),theta]),np.asarray([0.,cfg['duration_s']]),stats,dict(input_energy_J=float(sum(_power_at(s,cfg['duration_s']) for s in cfg['heat_sources'])),convective_energy_J=0.,energy_balance_error_J=residual,maximum_step_residual_J=residual,steady_residual=residual)

def make_slice(points,tets,frames,axis,value):
    eps=max(float(np.ptp(points,axis=0).max())*1e-10,1e-14)
    coord=points[:,axis]-value;selected=tets[(coord[tets].min(1)<=eps)&(coord[tets].max(1)>=-eps)]
    vertices=[];nodes=[];weights=[];faces=[];lookup={};other=[a for a in range(3) if a!=axis]
    for tet in selected:
        poly=[]
        for node in tet:
            if abs(coord[node])<=eps:
                key=(int(node),int(node))
                if key not in lookup:lookup[key]=len(vertices);vertices.append(points[node]);nodes.append(key);weights.append(0.)
                poly.append(lookup[key])
        for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
            a,b=sorted((int(tet[i]),int(tet[j])))
            if (coord[a]<-eps and coord[b]>eps) or (coord[b]<-eps and coord[a]>eps):
                key=(a,b)
                if key not in lookup:
                    w=-coord[a]/(coord[b]-coord[a]);lookup[key]=len(vertices);vertices.append(points[a]*(1-w)+points[b]*w);nodes.append(key);weights.append(w)
                poly.append(lookup[key])
        poly=list(dict.fromkeys(poly))
        if len(poly)>=3:
            q=np.array([vertices[p] for p in poly]);q-=q.mean(0)
            poly=np.asarray(poly)[np.argsort(np.arctan2(q[:,other[1]],q[:,other[0]]))]
            for j in range(1,len(poly)-1):faces.append([poly[0],poly[j],poly[j+1]])
    if not faces:return np.empty((0,3)),np.empty((0,3),dtype=np.uint32),np.empty((len(frames),0))
    nodes=np.asarray(nodes);w=np.asarray(weights);faces=np.asarray(faces)
    _,unique=np.unique(np.sort(faces,axis=1),axis=0,return_index=True);faces=faces[np.sort(unique)]
    values=frames[:,nodes[:,0]]*(1-w)+frames[:,nodes[:,1]]*w
    return np.asarray(vertices),faces,values

def export_slice(job,axis,value):
    if axis not in (0,1,2) or not np.isfinite(value):raise ValueError('无效剖面')
    key=f'{axis}_{value:.9g}'.replace('-','n').replace('.','p')
    folder=job/'slices'/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():return read_json(folder/'manifest.json')
    mesh=np.load(job/'mesh.npz');frames=np.load(job/'temperatures.npy',mmap_mode='r')
    lo,hi=mesh['points'][:,axis].min(),mesh['points'][:,axis].max()
    if value<lo or value>hi:raise ValueError('剖面位置超出工件范围')
    pts,faces,values=make_slice(mesh['points'],mesh['tets'],frames,axis,value)
    pts.astype('<f4').tofile(folder/'points.bin');faces.astype('<u4').tofile(folder/'faces.bin');values.astype('<f4').tofile(folder/'values.bin')
    meta=dict(key=key,vertices=len(pts),triangles=len(faces),frames=len(frames),axis=axis,value=value)
    write_json(folder/'manifest.json',meta);return meta

def export_archive(job,mesh,frames,times,labels):
    dest=job/'export';dest.mkdir(exist_ok=True)
    hf=h5py.File(dest/'thermal-fields.h5','w')
    hf.create_dataset('points',data=mesh['points'],compression='gzip',shuffle=True)
    hf.create_dataset('tetra',data=mesh['tets'].astype(np.int32),compression='gzip',shuffle=True)
    hf.create_dataset('material',data=labels,compression='gzip',shuffle=True)
    xml=['<?xml version="1.0"?>','<Xdmf Version="3.0"><Domain><Grid Name="Temperature" GridType="Collection" CollectionType="Temporal">']
    n=len(mesh['points']);ne=len(mesh['tets'])
    for i,t in enumerate(times):
        hf.create_dataset(f'temperature/{i}',data=frames[i],compression='gzip',shuffle=True)
        xml.append(f'<Grid GridType="Uniform"><Time Value="{t:g}"/><Topology TopologyType="Tetrahedron" NumberOfElements="{ne}"><DataItem Dimensions="{ne} 4" NumberType="Int" Precision="4" Format="HDF">thermal-fields.h5:/tetra</DataItem></Topology><Geometry GeometryType="XYZ"><DataItem Dimensions="{n} 3" NumberType="Float" Precision="8" Format="HDF">thermal-fields.h5:/points</DataItem></Geometry><Attribute Name="Temperature_C" AttributeType="Scalar" Center="Node"><DataItem Dimensions="{n}" NumberType="Float" Precision="4" Format="HDF">thermal-fields.h5:/temperature/{i}</DataItem></Attribute><Attribute Name="Material_ID" AttributeType="Scalar" Center="Cell"><DataItem Dimensions="{ne}" NumberType="Int" Precision="4" Format="HDF">thermal-fields.h5:/material</DataItem></Attribute></Grid>')
    hf.close();xml.append('</Grid></Domain></Xdmf>');(dest/'temperature.xdmf').write_text('\n'.join(xml),encoding='utf-8')
    with zipfile.ZipFile(job/'result.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('thermal-fields.h5','temperature.xdmf'):z.write(dest/name,name)
        for name in ('config.json','audit.json','history.csv'):z.write(job/name,name)

def solve_job(job):
    cfg=read_json(job/'config.json');folder=MODELS/cfg['model_id'];display=dict(np.load(folder/'display.npz'))
    progress(job,'running',5,'准备几何与材料分区')
    mesh=build_mesh(folder,cfg,job)
    np.savez_compressed(job/'mesh.npz',**mesh)
    if len(mesh['points'])*(cfg['duration_s']/cfg['save_s']+2)>25000000:raise ValueError('结果规模过大，请增大保存间隔或网格尺寸。')
    progress(job,'running',35,'组装多材料导热、热容量和表面边界')
    M,K,C,G,loads,labels,display_nodes,display_bary,audit,rad_area,rad_ambient,rad_eps,phase_entries=assemble(mesh,cfg,display)
    progress(job,'running',55,'开始'+('稳态' if cfg.get('analysis_mode')=='steady' else '瞬态')+'计算')
    callback=lambda fraction,msg:progress(job,'running',55+int(30*fraction),msg)
    if cfg.get('analysis_mode')=='steady':
        theta,times,stats,energy=steady(M,K,C,G,loads,cfg,callback,rad_area,rad_ambient,rad_eps)
    else:
        theta,times,stats,energy=integrate(M,K,C,G,loads,cfg,callback,rad_area,rad_ambient,rad_eps,phase_entries)
    # Report the actual volume average, distinct from the heat-capacity-weighted
    # average used internally for the energy balance in a heterogeneous solid.
    volume_weights=np.bincount(mesh['tets'].ravel(),weights=np.repeat(mesh['tet_volumes']/4,4),minlength=len(mesh['points']))
    for row,values in zip(stats,theta):row['average_C']=float(volume_weights@values/volume_weights.sum()+cfg['initial_C'])
    frames=theta+np.float32(cfg['initial_C']);np.save(job/'temperatures.npy',frames)
    surface=np.einsum('fvi,vi->fv',frames[:,display_nodes],display_bary).astype('<f4');surface.tofile(job/'surface.bin')
    audit.update(energy=energy,nodes=len(mesh['points']),tetrahedra=len(mesh['tets']),minimum_quality=float(mesh.get('minimum_quality',0)),stats=stats,
        material_interface='CAD-conforming box fragments' if read_json(folder/'metadata.json')['kind']=='step' else 'element-centroid assignment; refine mesh at material interfaces',
        assumptions=['constant material properties outside phase-change intervals','smeared interface resistance when configured','prescribed convection coefficient plus optional surface radiation; no airflow solve'])
    warnings=[]
    lower_bound=min([cfg['initial_C'],cfg['ambient_C']]+[c['ambient_C'] for c in cfg['cooling']])
    undershoot=lower_bound-float(frames.min())
    if undershoot>1e-4:warnings.append(f'最低温度比所有初始/环境温度低 {undershoot:.4f}°C，属于离散数值下冲；请调整网格与时间步长复核。结果未做截断修饰。')
    if float(mesh.get('minimum_quality',1))<.05:warnings.append('局部存在形状质量较低的网格单元，请通过网格加密比较确认热点精度。')
    for h in audit['heat_sources']:
        selected_area=h.get('selected_area_m2')
        relative=abs(h['mapped_area_m2']/selected_area-1) if selected_area else 0
        if relative>.05:warnings.append(f'热源“{h["name"]}”的网格面积与选区面积差 {relative:.1%}；总功率已保持，建议细化热源附近网格。')
    audit['warnings']=warnings
    write_json(job/'audit.json',audit)
    with (job/'history.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=stats[0].keys());writer.writeheader();writer.writerows(stats)
    manifest=dict(times_s=times.tolist(),vertices=surface.shape[1],frames=len(times),minimum_C=float(frames.min()),maximum_C=float(frames.max()),stats=stats,summary=audit)
    write_json(job/'result.json',manifest)
    progress(job,'running',88,'生成内部剖面和完整结果文件')
    bounds=np.array(read_json(folder/'metadata.json')['bounds_m']);export_slice(job,0,float(bounds[:,0].mean()))
    export_archive(job,mesh,frames,times,labels)
    progress(job,'completed',100,'计算完成',finished_at=time.time())
