"""Pure sysfs topology and bounded-memory contracts shared by R22 tools."""
import hashlib
import errno
from pathlib import Path
import re
import statistics


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_list(text):
    require(type(text) is str and text.strip(),'unknown topology: empty CPU list')
    values=[]
    for item in text.strip().split(','):
        require(re.fullmatch(r'\d+(?:-\d+)?',item),'unknown topology: malformed CPU list')
        bounds=list(map(int,item.split('-')))
        require(len(bounds)==1 or bounds[1]>=bounds[0],'unknown topology: reversed CPU range')
        values.extend([bounds[0]] if len(bounds)==1 else range(bounds[0],bounds[1]+1))
    require(len(values)==len(set(values)),'unknown topology: duplicate CPU list entries')
    return sorted(values)


def capture_topology(eligible,cpu_root=Path('/sys/devices/system/cpu')):
    require(type(eligible) is list and len(eligible)>=8 and all(type(v) is int and v>=0 for v in eligible)
        and len(eligible)==len(set(eligible)),'at least eight distinct eligible logical CPUs required')
    root=Path(cpu_root);files={};optional_files_absent={}
    def read(path,optional=False):
        try:data=path.read_bytes()
        except FileNotFoundError as error:
            if not optional or error.errno!=errno.ENOENT:raise ValueError('unknown topology: cannot read '+str(path)) from error
            canonical=str(path.resolve())
            require(canonical not in files,'topology optional file disappeared while reading')
            optional_files_absent[canonical]=dict(status='absent',error='ENOENT')
            return None
        except (OSError,UnicodeError) as error:raise ValueError('unknown topology: cannot read '+str(path)) from error
        text=data.decode('ascii').strip();canonical=str(path.resolve())
        value=dict(sha256=hashlib.sha256(data).hexdigest(),value=text)
        require(canonical not in optional_files_absent,'topology optional file appeared while reading')
        require(canonical not in files or files[canonical]==value,'topology changed while reading shared sysfs file')
        files[canonical]=value;return text
    def integer(path,optional=False):
        text=read(path,optional)
        if text is None:return None
        require(re.fullmatch(r'\d+',text),'unknown topology: nonnegative integer unavailable: '+str(path));return int(text)
    online=cpu_list(read(root/'online'))
    require(set(eligible)<=set(online),'unknown topology: eligible CPU is offline')
    def cache_key(package,llc):
        # Optional kernel cache IDs never select or sort structural domains.
        return package,llc['level'],llc['type'],tuple(llc['shared_cpu_list'])
    def last_cache(cpu):
        directory=root/('cpu'+str(cpu))
        caches=[]
        try:indices=sorted(path for path in (directory/'cache').iterdir() if re.fullmatch(r'index\d+',path.name))
        except OSError as error:raise ValueError('unknown topology: cannot list cache indices: '+str(directory)) from error
        for cache in indices:
            level=integer(cache/'level');kind=read(cache/'type')
            if kind not in ('Data','Unified'):continue
            identity=integer(cache/'id',optional=True);shared=cpu_list(read(cache/'shared_cpu_list'))
            require(cpu in shared,'unknown topology: CPU absent from cache shared list')
            require(set(shared)<=set(online),'unknown topology: cache sharing includes offline/unknown CPU')
            caches.append(dict(level=level,id=identity,id_status='absent' if identity is None else 'present',shared_cpu_list=shared,type=kind))
        require(caches,'unknown topology: last-level data/unified cache unavailable')
        maximum=max(row['level'] for row in caches);last=[row for row in caches if row['level']==maximum]
        signatures={(row['type'],tuple(row['shared_cpu_list'])) for row in last}
        require(len(signatures)==1,'unknown topology: ambiguous last-level cache')
        ids={row['id'] for row in last if row['id'] is not None}
        require(len(ids)<=1,'unknown topology: ambiguous last-level cache IDs')
        llc=dict(last[0]);llc.update(id=next(iter(ids)) if ids else None,id_status='present' if ids else 'absent')
        return llc
    rows=[];peers={}
    for cpu in sorted(eligible):
        directory=root/('cpu'+str(cpu));topology=directory/'topology'
        package=integer(topology/'physical_package_id');core=integer(topology/'core_id')
        siblings=cpu_list(read(topology/'thread_siblings_list'))
        require(cpu in siblings,'unknown topology: CPU absent from thread siblings')
        llc=last_cache(cpu);peers[cpu]=dict(cpu=cpu,package=package,llc=llc)
        nodes=[node for node in directory.glob('node[0-9]*') if re.fullmatch(r'node\d+',node.name)]
        require(len(nodes)==1,'unknown topology: CPU must belong to exactly one NUMA node')
        node=int(nodes[0].name[4:]);node_cpus=cpu_list(read(nodes[0]/'cpulist'))
        require(cpu in node_cpus,'unknown topology: CPU absent from NUMA cpulist')
        rows.append(dict(cpu=cpu,package=package,core=core,thread_siblings=siblings,
            numa_node=node,numa_cpus=node_cpus,llc=llc))
    # Read every advertised shared peer, even outside the eligible affinity budget.
    # These rows only prove LLC identity and can never become selected CPUs.
    pending={cpu for row in peers.values() for cpu in row['llc']['shared_cpu_list']}-set(peers)
    while pending:
        cpu=min(pending);pending.remove(cpu)
        package=integer(root/('cpu'+str(cpu))/'topology/physical_package_id')
        llc=last_cache(cpu);peers[cpu]=dict(cpu=cpu,package=package,llc=llc)
        pending.update(set(llc['shared_cpu_list'])-set(peers))
    domain_ids={}
    for row in peers.values():
        key=cache_key(row['package'],row['llc'])
        for cpu in row['llc']['shared_cpu_list']:
            other=peers[cpu]
            require(cache_key(other['package'],other['llc'])==key,
                    'unknown topology: shared LLC peers disagree on cache domain identity/package')
        ids=domain_ids.setdefault(key,set())
        if row['llc']['id'] is not None:ids.add(row['llc']['id'])
        require(len(ids)<=1,'unknown topology: shared LLC peers disagree on readable cache IDs')
    by_cpu={row['cpu']:row for row in rows};physical={}
    for row in rows:
        for cpu in set(row['numa_cpus'])&set(eligible):
            other=by_cpu[cpu]
            require(other['numa_node']==row['numa_node'] and other['numa_cpus']==row['numa_cpus'],
                    'unknown topology: shared NUMA peers disagree on node domain identity')
        key=(row['package'],row['core']);physical.setdefault(key,[]).append(row)
        for sibling in set(row['thread_siblings'])&set(eligible):
            other=by_cpu[sibling]
            require((other['package'],other['core'])==key and other['thread_siblings']==row['thread_siblings'],
                    'unknown topology: inconsistent physical core/thread siblings')
    representatives=[]
    for key,group in physical.items():
        first=group[0]
        require(all(row['thread_siblings']==first['thread_siblings'] and row['numa_node']==first['numa_node'] and
                    cache_key(row['package'],row['llc'])==cache_key(first['package'],first['llc']) for row in group),
                'unknown topology: SMT siblings disagree on core/LLC/NUMA')
        representatives.append(min(group,key=lambda row:row['cpu']))
    choices=[]
    for node in sorted({row['numa_node'] for row in representatives}):
        candidates=[row for row in representatives if row['numa_node']==node]
        if len(candidates)<8:continue
        groups={}
        for row in candidates:
            llc=row['llc'];key=cache_key(row['package'],llc)
            groups.setdefault(key,[]).append(row['cpu'])
        ordered=sorted(groups,key=lambda key:(-len(groups[key]),key))
        chosen=[];used=[]
        for key in ordered:
            chosen.extend(sorted(groups[key])[:8-len(chosen)]);used.append(key)
            if len(chosen)==8:break
        choices.append((len(used),node,sorted(chosen),used))
    require(choices,'eligible set has no eight different physical cores within one NUMA node')
    count,node,selected,used=min(choices,key=lambda row:(row[0],row[1],row[2]))
    return dict(schema=1,cpu_root=str(root),eligible_cpus=sorted(eligible),selected_cpus=selected,
        online_cpus=online,cpus=rows,files=dict(sorted(files.items())),
        optional_files_absent=dict(sorted(optional_files_absent.items())),
        llc_peers=[peers[cpu] for cpu in sorted(peers)],
        choice=dict(physical_cores=8,numa_node=node,llc_count=count,
            identity_method='physical package, level, type and complete reciprocal shared CPU domain; optional readable IDs must agree',
            policy='minimum LLC count among eight distinct physical cores in one NUMA node; single LLC first',
            tie_break='largest LLC core groups first, then NUMA ID and CPU IDs within the declared eligible set',
            llcs=[dict(package=key[0],level=key[1],type=key[2],shared_cpu_list=list(key[3]),
                id=next(iter(domain_ids[key])) if domain_ids[key] else None) for key in used]),
        boundary='sysfs topology and inherited affinity budget; no instantaneous scheduler trace or load ranking')


def verify_topology(receipt,selected,cpu_root=None):
    require(type(receipt.get('schema')) is int and receipt['schema']==1 and receipt.get('selected_cpus')==list(selected),
            'shared topology receipt selection mismatch')
    actual=capture_topology(receipt.get('eligible_cpus'),cpu_root or Path(receipt['cpu_root']))
    require(actual==receipt,'physical CPU/LLC/NUMA topology changed or receipt was altered')
    return actual


def validate_memory_trend(checkpoints):
    require(type(checkpoints) is list and len(checkpoints)==20 and
        [row.get('completed') for row in checkpoints]==list(range(1000,20001,1000)),
        'stability requires all twenty fixed 1000-forward checkpoints')
    rss=[];allocated=[];reserved=[]
    for row in checkpoints:
        memory=row.get('memory',{});allocator=memory.get('npu_allocator',{})
        for field in ('rss_kib','whole_process_peak_rss_kib'):
            require(type(memory.get(field)) is int and memory[field]>=0,'stability RSS/HWM unavailable')
        require(allocator.get('available') is not False,'stability NPU allocator evidence unavailable')
        for field in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
            require(type(allocator.get(field)) is int and allocator[field]>=0,'stability NPU allocator metric missing: '+field)
        rss.append(memory['rss_kib']);allocated.append(allocator['memory_allocated']);reserved.append(allocator['memory_reserved'])
    def comparison(values):
        early=statistics.median(values[:5]);late=statistics.median(values[-5:])
        return dict(first_five_median=early,last_five_median=late,growth=late-early)
    host,device,pool=comparison(rss),comparison(allocated),comparison(reserved)
    require(host['growth']<=64*1024,'stability RSS median growth exceeds 64MiB')
    require(device['growth']<=16*1024**2,'stability allocated median growth exceeds 16MiB')
    return dict(passed=True,rss_kib=host,npu_allocated_bytes=device,npu_reserved_bytes=pool,
        rss_limit_kib=64*1024,allocated_limit_bytes=16*1024**2,reserved_threshold_applied=False,
        boundary='bounded 20000-forward exposure; whole-process HWM and allocator peaks; no universal leak/stability guarantee')
