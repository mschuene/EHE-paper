
#!/bin/bash
cd "$(dirname "$0")"
mkdir EHETorus_crit
cd EHETorus_crit
jobid=$(qsub -terse -N EHETorus_crit -q `cat /0/maik/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-360 <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
export OMP_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen
python <<EOF
# common setup code
import matplotlib
matplotlib.use('Agg')
import pickle 
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/EHETorus_crit"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ,flush=True)
task_id = int(os.environ.get('SGE_TASK_ID'))
from ehe import * 
from ehe_ana import * 
from utils import * 
from plfit_cd import * 
from ehe_subnetworks import *

from ehe import *;
parameters = [(k,alpha_idx,L) for k in [1,2,3] for L in [200,100,300] for alpha_idx in range(40)]
#parameters = [(k,alpha_idx,L) for k in [1] for L in [100] for alpha_idx in range(50)]

k,alpha_idx,L = parameters[task_id -1]

#alpha = np.logspace(np.log10(0.95),np.log10(0.999),40)[alpha_idx]
alpha = np.logspace(np.log10(0.999),np.log10(0.999999),20)[alpha_idx]
alpha = alpha/((2*k+1)**2-1)        
conn = get_conn_list_k(L,k)
deltaU = 1-((2*k+1)**2-1)*alpha - 1e-4         
        
e = load_module('ehe_detailed').EHE()
e.simulate_model_grid(np.random.random(int(conn.shape[0])),int(1e6),alpha,conn,deltaU)
av = e.get_avs_size_and_duration();
sps = e.get_spiking_patterns();
step2 = np.array([len(sp[1]) if len(sp)> 1 else 0 for sp in sps])
pickle.dump([av,np.mean(step2)],open(outdir+'/ehetorus_'+str(L)+'_'+str(k)+'_'+str(40+alpha_idx)+'.pickle','wb'))

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessEHETorus_crit -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
