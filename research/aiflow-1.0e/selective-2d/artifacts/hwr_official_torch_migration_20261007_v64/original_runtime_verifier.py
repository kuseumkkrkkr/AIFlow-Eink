import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
import scipy
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--context', required=True)
parser.add_argument('--output', required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
assert Path(sys.prefix).resolve()==root,(sys.prefix,root)
assert Path(torch.__file__).resolve().is_relative_to(root),torch.__file__
assert torch.__version__=='2.14.1+cpu',torch.__version__
assert torch.version.cuda is None
x=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
y=x@x.T
y.sum().backward()
assert torch.equal(y,torch.tensor([[5.,11.],[11.,25.]]))
assert torch.equal(x.grad,torch.tensor([[8.,12.],[8.,12.]]))
assert torch.equal(torch.from_numpy(np.array([1,2,3],dtype=np.float32))*2,torch.tensor([2.,4.,6.]))
pip_check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
assert pip_check.returncode==0,pip_check.stdout+pip_check.stderr
report={'status':'passed','verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_context':args.context,'hostname':socket.gethostname(),'python_executable':sys.executable,'python_version':sys.version,'prefix':sys.prefix,'torch_version':torch.__version__,'torch_path':torch.__file__,'numpy_version':np.__version__,'numpy_path':np.__file__,'scipy_version':scipy.__version__,'scipy_path':scipy.__file__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'matrix_product':y.tolist(),'gradient':x.grad.tolist(),'numpy_interoperability':'passed','pip_check':pip_check.stdout.strip()}
output=Path(args.output).resolve()
assert output.is_relative_to(root),output
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
