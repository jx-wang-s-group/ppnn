import numpy as np
import os
import shutil
from tqdm import tqdm
from io import StringIO

def readonestep(stepdir):

    with open(os.path.join(stepdir,'p'), 'r') as f:
        contents = f.readlines()
        num_mesh = int(contents[19])
        p = np.loadtxt(contents,skiprows=21,max_rows=num_mesh)
   

    with open(os.path.join(stepdir,'U'), 'r') as f:
        contents = f.readlines()
        content = [i[1:-3] for i in contents[21:]]
        u = np.loadtxt(content,max_rows=num_mesh)

    return p,u


def readall(dir):
    presults = []
    uresults = []
    times = sorted(os.listdir(dir))
    os.chdir(dir)
    try:
        times.remove('0')
    except:
        pass
    for i in tqdm(times):
        try:
            float(i)
        except:
            continue
        p,u=readonestep(i)
        presults.append(p)
        uresults.append(u)
        # print('step: {0} read'.format(i))

    return np.stack(presults,axis=0),np.stack(uresults,axis=0)


def map2coarse(src,dst,t,len_stp=0.001):
    tmpdir = 'coarse_tmp'
    oldpath = os.getcwd()
    # try:
    #     shutil.copytree('coarse_template',tmpdir)
    # except FileExistsError:
    #     pass
    # os.chdir(dst)
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
    \\\  /    A nd           | Version:  9
     \\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
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


utail = """\
)
;

boundaryField
{
    movingWall
    {
        type            fixedValue;
        value           uniform (100 0 0);
    }
    fixedWalls
    {
        type            noSlip;
    }
    frontAndBack
    {
        type            empty;
    }
}


// ************************************************************************* //
"""

def pheader(n):
    return """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  9
     \\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<vector>\n"""+\
    str(n)+'\n'+\
    '(\n'


ptail = """\
)
;

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }
    fixedWalls
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}


// ************************************************************************* //"""


def writeofvec(u, dir):
    
    num_mesh = u.shape[0]
    u = np.concatenate([u,np.zeros((num_mesh,1))],axis=1)
    middleio = StringIO()
    np.savetxt(middleio,u,fmt='(%.6g %.6g %.0g)')

    towrite = uheader(num_mesh) + middleio.getvalue() + utail

    with open(os.path.join(dir,'U'),'w+') as f:
        f.write(towrite)
        

def writeofsca(p, dir):

    num_mesh = p.shape[0]
    
    middleio = StringIO()
    np.savetxt(middleio,p,fmt='%.6g')

    towrite = pheader(num_mesh) + middleio.getvalue() + ptail

    with open(os.path.join(dir,'p'),'w+') as f:
        f.write(towrite)

if __name__=='__main__':
    # import sys
    
    # p,u=readall(sys.argv[1])

    # np.save('pressurec.npy',p)
    # np.save('velocityc.npy',u)

    # import matplotlib.pyplot as plt
    # plt.pcolormesh(p[2].reshape(400,400))
    # plt.show()
    # plt.pcolormesh(np.sqrt(u[2,:,0]**2+u[2,:,1]**2).reshape(400,400))
    # plt.show()
    # for t in range(1078):
    #     map2coarse('10000','./coarse3',t/1000)
    # map2coarse('10000','coarse3',0.02)
    u = np.linspace(0,99,100)
    # u = np.stack([u/100,-u/100],axis=-1)
    writeofsca(u,'./')
    

    




