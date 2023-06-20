
1+3

1+1

import time
import os
import fcntl
import errno

class SimpleFlock:
   """Provides the simplest possible interface to flock-based file locking. Intended for use with the `with` syntax. It will create/truncate/delete the lock file as necessary."""

   def __init__(self, path, timeout = None):
      self._path = path
      self._timeout = timeout
      self._fd = None

   def __enter__(self):
      self._fd = os.open(self._path, os.O_CREAT)
      start_lock_search = time.time()
      while True:
         try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Lock acquired!
            return
         except (OSError, IOError) as ex:
            if ex.errno != errno.EAGAIN: # Resource temporarily unavailable
               raise
            elif self._timeout is not None and time.time() > (start_lock_search + self._timeout):
               # Exceeded the user-specified timeout.
               raise

         # TODO It would be nice to avoid an arbitrary sleep here, but spinning
         # without a delay is also undesirable.
         time.sleep(0.1)

   def __exit__(self, *args):
      fcntl.flock(self._fd, fcntl.LOCK_UN)
      os.close(self._fd)
      self._fd = None

      # Try to remove the lock file, but don't try too hard because it is
      # unnecessary. This is mostly to help the user see whether a lock
      # exists by examining the filesystem.
      try:
         os.unlink(self._path)
      except:
         pass

import sys
sys.path.insert(0,'./cpp')
import cppimport
from time import gmtime, strftime
import sys
from cppimport.find import find_module_cpppath
from cppimport.importer import setup_module_data
from cppimport.checksum import is_checksum_current
import os.path
import os
def load_module(fullname):
    name = find_module_cpppath(fullname)
    hostname = os.uname()[1].replace("-","_")
    suffix = strftime("%Y%m%d%H%M%S",gmtime())
    hostspezsuff = hostname+'__spez__'+suffix
    with open(name) as f:
        content = f.read()       
    (d,fname) = os.path.split(name)
    bfname,ext = os.path.splitext(fname)
    print('now try acquire lock',flush=True)
    with SimpleFlock('/tmp'+os.sep+hostname+'flock'):
        print('lock acquired',flush=True)
        current_suffixes = [os.path.splitext(f)[0].split(hostname+'__spez__')[1]  for f in
                            os.listdir(d) if f.startswith(bfname+hostname+'__spez__') and
                                             f.endswith(ext)]
        if len(current_suffixes) > 0:
            oldhostspezsuff = hostname+'__spez__'+current_suffixes[0]
            oldhostspezfname = bfname+oldhostspezsuff+ext
            module_data = setup_module_data(fullname+oldhostspezsuff,d+os.sep+oldhostspezfname)
            oldsuffixed_content = content.replace("__module_suffix__",oldhostspezsuff)
            with open(d+os.sep+oldhostspezfname,'w') as oldf:
                oldf.write(oldsuffixed_content)
            print('old_suff',oldhostspezsuff,flush=True)
            if is_checksum_current(module_data):
                print('dont recompile, call imp with '+str(fullname+oldhostspezsuff),flush=True)
                return cppimport.imp(fullname+oldhostspezsuff)
            else:
                print('recompile needed',flush=True)
                for ftd in [module_data['ext_path'],module_data['filepath'],
                            d+os.sep+'.rendered.'+module_data['fullname']+ext,
                            d+os.sep+'.'+module_data['fullname']+ext+'.cppimporthash']:
                    try:
                        print('try removing',ftd,flush=True)
                        os.remove(ftd)
                    except:
                        pass
        suffixed_content = content.replace("__module_suffix__",hostspezsuff)
        with open(d+os.sep+bfname+hostspezsuff+ext,'w') as fhs:
            fhs.write(suffixed_content)
        print('call imp with '+str(fullname+hostspezsuff))     
        return cppimport.imp(fullname+hostspezsuff)

from mako.template import Template
from mako.runtime import Context
from io import StringIO
from os.path import basename
from os.path import splitext

def render_cluster_template(**opts):
    default_options={'queue':'long_64gb',
                     'mail':'maikschuenemann@gmail.com',
                     'output':'o',
                     'error':'e',
                     'priority':0,
                     'task_ids':'1',
                     'content':"print('hi')",
                     'max_threads':1,
                     'postprocessing_content':"",
                     'interpreter':'python',
                     'template_file':"./util/cluster_template.sh",
                     'output_file':None}
    options = default_options.copy()
    options.update(opts)
    if 'name' not in options and options['output_file'] is not None:
        options['name'] = splitext(basename(options['output_file']))[0]
    template = Template(strict_undefined=True,filename=options['template_file'])
    output = StringIO()
    ctx = Context(output,**options)
    template.render_context(ctx)
    if options['output_file'] is not None:
        with open(options['output_file'],"w") as f:
            f.write(output.getvalue())
    return output.getvalue()

import sh
def ssh_cluster(servername='server_inline'):
    return sh.ssh.bake("-t",servername,
    "export GE_CELL=neuro;export SGE_ROOT=/sge-root; export SGE_CLUSTER_NAME=OurCluster;cd /home/maik/master/src;")

server= ssh_cluster('server')
server_inline = ssh_cluster()
badweather = ssh_cluster('maik@badweather')

def rsync_server(server='server_inline'):
    print('rsyncing server')
    print(sh.rsync('-rtvu','./',server+":/home/maik/master/src/"))
    #print(sh.rsync('-rtvu','../avalanches/',server+":/home/maik/master/avalanches/"))
    print('done')

def qsub(command="print('hi')",post_command='',name=None,outfile=None,servername='server_inline',execute=True,**options):
    ssh = ssh_cluster(servername)
    options['content'] = command
    options['postprocessing_content']=post_command
    if name is not None:
        options['name'] = name
        if outfile is None:
            outfile = "./cluster/"+name+".sh"
            options['output_file'] = outfile
    elif outfile is not None:
        options['output_file'] = outfile
    else:
        raise Exception('must be called with either name or outfile')
    rendered = render_cluster_template(**options)
    rsync_server(servername)
    if execute:
        print(ssh("sh "+options['output_file']))

def scp_result(filename,servername='server_inline'):
   outfile = '../avalanches/'+filename 
   try:
      os.makedirs(os.path.dirname(outfile))
   except:
        pass
   print(sh.scp(servername+':/home/maik/master/avalanches/'+filename,outfile))

import os
from os.path import basename
from os.path import splitext
import numpy as np
def post_command_concat(prefix,tid_range,axis=0):
    dname,bname = os.path.split(prefix)
    fname = splitext(bname)[0]
    arrays = [];
    failed = "";
    for tid in tid_range:
        try:
            arrays.append(np.load(os.path.join(dname,fname+str(tid)+".npy")))
        except:
            failed += str(tid)+"\n"
    post_array = np.concatenate(arrays,axis=axis)
    np.save(os.path.join(dname,fname+"concatenated.npy"),post_array)
    with open(os.path.join(dname,'failed_idx.txt'),'w') as f:
        f.write(failed)

import os
from os.path import basename
from os.path import splitext
from os.path import join
import numpy as np

def concat_detailed(avs_prefix,avs_inds_prefix,tid_range):
    dname,bname = os.path.split(avs_prefix)
    fname = splitext(bname)[0]
    fname_inds = splitext(basename(avs_inds_prefix))[0]
    concatenated_avs = []
    concatenated_avs_inds = []
    failed = ""
    for tid in tid_range:
        try:
            avs_tid = np.load(join(dname,fname+str(tid)+".npy"))
            avs_inds_tid = np.load(join(dname,fname_inds+str(tid)+".npy")) + len(concatenated_avs)
            concatenated_avs.extend(avs_tid)
            concatenated_avs_inds.extend(avs_inds_tid)
        except:
            import traceback
            traceback.print_exc()
            failed += str(tid)+"\n"
    concatenated_avs = np.array(concatenated_avs)
    concatenated_avs_inds = np.array(concatenated_avs_inds)
    np.save(join(dname,fname+"concatenated.npy"),concatenated_avs)
    np.save(join(dname,fname_inds+"concatenated.npy"),concatenated_avs_inds)
    with open(os.path.join(dname,'failed_idx.txt'),'w') as f:
        f.write(failed)
    return (concatenated_avs,concatenated_avs_inds)

import matplotlib.pyplot as plt
import numpy as np

    
def spike_raster(spike_trains,ax=None,markersize=5):
    if ax is None:
        ax = plt.gca();
    for i, t in enumerate(spike_trains):
        plt.plot(t, i * np.ones_like(t), 'k.', markersize=markersize)
    ax.axis('tight')
    ax.tick_params(axis='both', which='major', labelsize=14)
    return ax

def sym_kl(pdf,pl_exp,upper_limit=None,lower_limit=None):
    pdf_emp,unq_points = pdf
    lower_limit = unq_points[0] if lower_limit is None else lower_limit
    upper_limit = unq_points[-1] if upper_limit is None else upper_limit
    pdf = np.zeros(upper_limit - lower_limit + 1)
    for pe,up in zip(pdf_emp,unq_points): 
        if 0 <= up - lower_limit < len(pdf):
            pdf[up - lower_limit] = pe
    pdf[pdf != 0] /= np.sum(pdf)        
    pdf_pl,unq_true = discrete_power_law_dist(pl_exp,
                                              lower_limit=lower_limit,
                                              upper_limit=upper_limit)
    tmp = (pdf - pdf_pl)*(np.log(pdf)-np.log(pdf_pl))
    tmp[~np.isfinite(tmp)] = 0
    return np.nansum(tmp)

def sym_kl2(pdf1,pdf2,upper_limit=None,lower_limit=None): 
    pdf1_emp,unq_points1 = pdf1
    pdf2_emp,unq_points2 = pdf2         
    lower_limit = min(unq_points1[0],unq_points2[0]) if lower_limit is None else lower_limit
    upper_limit = max(unq_points1[-1],unq_points2[-1]) if upper_limit is None else upper_limit
    # populate distribution vectors
    pdf1 = np.zeros(upper_limit - lower_limit + 1)
    for pe1,up1 in zip(pdf_emp1,unq_points1):
        if 0 <= up1 - lower_limit < len(pdf):
            pdf1[up1 - lower_limit] = pe         
    pdf2 = np.zeros(upper_limit - lower_limit + 1)
    for pe2,up2 in zip(pdf_emp2,unq_points2):
        if 0 <= up2 - lower_limit < len(pdf):
            pdf2[up1 - lower_limit] = pe
    tmp = (pdf1 - pdf2)*(np.log(pdf1)-np.log(pdf2))
    tmp[~np.isfinite(tmp)] = 0
    return np.nansum(tmp)    

def sym_kl_old(pdf,pl_exp,N):
    pdf,unq_points = pdf
    pdf_pl,unq_true = discrete_power_law_dist(pl_exp,lower_limit=1,upper_limit=unq_points[-1])
    tmp = (pdf - pdf_pl)*(np.log(pdf)-np.log(pdf_pl))
    tmp[~np.isfinite(tmp)] = 0
    return np.nansum(tmp)

def discrete_power_law_dist(pl_exp,lower_limit=1,upper_limit=10000):
    """limits are inclusive"""
    unique = list(range(lower_limit,upper_limit+1))
    pdf_pl = np.array([np.power(l,pl_exp) for l in unique])
    return (pdf_pl/np.sum(pdf_pl),np.array(unique))

def kl(pdf,pl_exp,N):
    pdf,unq_points = pdf
    pdf_pl,unq_true = discrete_power_law_dist(pl_exp,lower_limit=1,upper_limit=unq_points[-1])
    tmp = pdf_pl*(np.log(pdf)-np.log(pdf_pl))
    tmp[~np.isfinite(tmp)] = 0
    return -np.nansum(tmp)
