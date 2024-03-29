import numpy as np
import torch
from io import StringIO
import pandas as pd
import os
import shutil
from .utils import mesh_convertor, numpy2string


def readonestep(stepdir):
    with open(os.path.join(stepdir,'p'), 'r') as f:
        num_mesh = int(f.readlines()[20])
    with open(os.path.join(stepdir,'p'), 'r') as f:
        p = pd.read_csv(f, skiprows=21, nrows=num_mesh, header=0, delim_whitespace=True,dtype=np.float32).to_numpy()
    

    with open(os.path.join(stepdir,'U'), 'r') as f:
        contents = f.readlines()

    content = [i[1:-3] for i in contents[22:]]
    ff = StringIO('\n'.join(content))
    u = pd.read_csv(ff, nrows=num_mesh, header=None, delim_whitespace=True, dtype=np.float32).to_numpy()
    return p,u


def readall(dir):
    presults = []
    uresults = []
    times = os.listdir(dir)
    os.chdir(dir)
    try:
        times.remove('0')
    except:
        pass
    for i in times:
        try:
            float(i)
        except:
            times.remove(i)
    timeID = np.array(times,dtype=float)
    timeID = timeID.argsort()
    for t in timeID:
        print(times[t])
        p,u=readonestep(times[t])
        presults.append(p)
        uresults.append(u)
        # print('step: {0} read'.format(i))

    return np.stack(presults,axis=0),np.stack(uresults,axis=0)


def map2coarse(src,dst,t,len_stp=0.001):
    tmpdir = 'coarse_tmp'
    oldpath = os.getcwd()
    try:
        shutil.copytree(src, tmpdir)
    except FileExistsError:
        pass
    os.chdir(dst)
    print('map field {0} ...'.format(t))

    assert os.system(
        'mapFields {0} -consistent -sourceTime {2} -case {1} >> /dev/null'\
            .format(src,tmpdir,t)) == 0
    

    os.chdir(tmpdir)

    print('icoFoam solving {0} ...'.format(t))
    assert os.system('icoFoam >> /dev/null') == 0
    print('icoFoam time {0} finished!'.format(t))


    os.chdir(oldpath)
    shutil.move(os.path.join(tmpdir,str(len_stp)),os.path.join(dst,'{0:.3f}'.format(t+len_stp)))



def uheader(n):
    return """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  8
     \\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>\n"""+\
    str(n)+'\n'+\
    '(\n'


def utail(pos,t,num_bcpoints=25):
    def ubc(num_mesh,pos,t):
        x = (np.linspace(0,num_mesh-1,num_mesh)+0.5)/num_mesh
        u = np.exp(-50*(x-pos)*(x-pos))
        v = np.sin(t)*np.exp(-50*(x-pos)*(x-pos))
        return np.stack((u,v,np.zeros_like(u)),axis=1)
    bcvalue = numpy2string(ubc(num_bcpoints,pos,t))
    return """\
)
;

boundaryField
{
    walls
    {
        type            noSlip;
    }
    inlet
    {
        type            groovyBC;
        refValue        nonuniform List<vector>\n"""+\
    str(num_bcpoints)+'\n'+\
    '(\n'+\
    bcvalue+\
    '\n)\n;\n'+\
"""\
        refGradient     uniform (0 0 0);
        valueFraction   uniform 1;
        value           nonuniform List<vector>\n"""+\
    str(num_bcpoints)+'\n'+\
    '(\n'+\
    bcvalue+\
    '\n)\n;\n'+\
"""\
        valueExpression "vector(exp(-50*(pos().y-{0})*(pos().y-{0})),sin(time())*pos().y/pos().y*exp(-50*(pos().y-{0})*(pos().y-{0})),0)";""".format(pos)+\
"""
        gradientExpression "vector(0,0,0)";
        fractionExpression "1";
        evaluateDuringConstruction 0;
        cyclicSlave     0;
        variables       "";
        timelines       (
);
        lookuptables    (
);
        lookuptables2D  (
);
    }
    outlet
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}


// ************************************************************************* //"""



def pheader(n):
    return """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  8
     \\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<scalar>\n"""+\
    str(n)+'\n'+\
    '(\n'


ptail="""\
)
;

boundaryField
{
    walls
    {
        type            zeroGradient;
    }
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    frontAndBack
    {
        type            empty;
    }
}


// ************************************************************************* //
"""


def viscosity(mu, m=None):
    transportProperties = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  8
     \\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel Newtonian ;\n"""+\
'nu              nu [0 2 -1 0 0 0 0] {0:.5g};\n'.format(mu)
    if m !=None: transportProperties += 'm               [0 0 -1 0 0 0 0] {0:.3g};\n\n'.format(m)
    transportProperties +="""
// ************************************************************************* //
"""
    return transportProperties





def writeofvec(u, dir, pos, t, inletpoints):
    
    num_mesh = u.shape[0]

    u = np.concatenate([u,np.zeros((num_mesh,1))],axis=1)

    ustring = numpy2string(u)

    towrite = uheader(num_mesh) + ustring + utail(pos,t,inletpoints)

    with open(os.path.join(dir,'U'),'w+') as f:
        f.write(towrite)
        

def writeofsca(p, dir):

    num_mesh = p.shape[0]
    
    pstring = numpy2string(p,'%.6g')

    towrite = pheader(num_mesh) + pstring + ptail

    with open(os.path.join(dir,'p'),'w+') as f:
        f.write(towrite)


def writeofvis(mu, dir, m=None):
    towrite = viscosity(mu, m)
    with open(os.path.join(dir,'transportProperties'),'w+') as f:
        f.write(towrite)


class OneStepRunOFCoarse(object):
    def __init__(self, template_path, tmp_path, dt, cmesh, 
                pos:float, mu:float, num_inletpoints:float,
                solver = 'icoFoam', m = 0.08) -> None:
        super().__init__()
        try:
            shutil.copytree(template_path,tmp_path)
        except FileExistsError:
            pass
        self.tmp_path = tmp_path
        self.dt = dt
        self.cmesh = cmesh
        self.pos = pos
        self.mu = mu
        self.num_inletpoints = num_inletpoints
        self.solver = solver
        if solver == 'icoFoam': m =None
        writeofvis(self.mu, os.path.join(self.tmp_path, 'constant'), m)
        with open(os.path.join(self.tmp_path, 'system/controlDict'),'r') as f:
            self.controlDict = f.readlines()
        
            
    def __call__(self, u0:torch.Tensor, t) -> torch.Tensor:
        
        error = False
        
        p = u0[0,2].reshape(1,-1).permute(1,0)
        u = u0[0,:2].reshape(2,-1).permute(1,0)
        
        t0s = '{0:3g}'.format(t).replace(' ','')
        t1s = '{0:3g}'.format(t+self.dt).replace(' ','')

        U0path = os.path.join(self.tmp_path, t0s)
        U1path = os.path.join(self.tmp_path, t1s)

        if not os.path.exists(U0path): os.makedirs(U0path)

        writeofvec(u.numpy(), U0path, self.pos,t,self.num_inletpoints)
        writeofsca(p.numpy(), U0path)
        
        # write controlDict
        self.controlDict[21] = 'startTime\t{0};\n'.format(t0s)
        self.controlDict[25] = 'endTime\t{0};\n'.format(t1s)
        with open(os.path.join(self.tmp_path, 'system/controlDict'),'w') as f:
            f.writelines(self.controlDict)


        oldpath = os.getcwd()
        os.chdir(self.tmp_path)
        try:
            ofreturn=os.system('{0} > Foam.log 2>&1'.format(self.solver))
        except:
            error = True
        if ofreturn != 0:
            error = True
        os.chdir(oldpath)
        if error:
            return float('nan')*torch.ones_like(u0), error
        else:
            p,u = readonestep(U1path)
            p = torch.from_numpy(p).float()
            u = torch.from_numpy(u).float()
            p = p.reshape(1,1,*self.cmesh)
            u = u.permute(1,0).reshape(2,*self.cmesh).unsqueeze(0)
            return torch.cat([u,p],dim=1),error







if __name__=='__main__':
    
    res = [2,4,6,8,10]
    ps = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]
    mscvter = mesh_convertor((100,400),(25,100),dim=2,align_corners=False)
    fdata = torch.load('/home/lxy/store/projects/dynamic/PDE_structure/OpenFoam/pipefine.pt'
        ).float()
    result = []
    for pos in range(len(ps)):
        p_result = []
        for re in range(len(res)):
            csolver = OneStepRunOFCoarse('/home/lxy/store/projects/dynamic/PDE_structure/OpenFoam/test',
                '/home/lxy/store/projects/dynamic/PDE_structure/OpenFoam/tmp',0.8,(25,100),ps[pos],0.001/res[re]
                ,25)
            r_result = []
            if pos==8 and re==4:
                continue
            phy_time = 15.2
            for t in range(56):
                u,error = csolver(mscvter.down(fdata[pos*len(res)+re,t:t+1]), phy_time)
                phy_time += 0.8
                u = mscvter.up(u)
                r_result.append(u)
                if error:
                    print('{0} pos,{1} Re,{2} time, error'.format(ps[pos],res[re],phy_time))
                    raise Exception('error')
            r_result = torch.cat(r_result,dim=0)
            p_result.append(r_result)
            print('{0} pos,{1} Re, done'.format(ps[pos],res[re]))
        p_result = torch.stack(p_result,dim=0)
        result.append(p_result)
    result = torch.cat(result,dim=0)
    print(result.shape)
    torch.save(result,'/home/lxy/store/projects/dynamic/PDE_structure/OpenFoam/pipeflow_more_coarse.pt')

                

    




