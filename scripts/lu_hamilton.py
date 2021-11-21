import numpy as np
from astropy.table import Table, Column


def check_neighbors(i,j,k, fill_arr, fill_value):
    # Checks the nearest neighbors for a given array, fill_arr
    # and fills in array with the fill_value array
    
    if i < fill_arr.shape[0]-1:fill_arr[i+1,j,k]=fill_value[0]
    if i > 0: fill_arr[i-1,j,k]=fill_value[0]
    
    if j < fill_arr.shape[1]-1:fill_arr[i,j+1,k]=fill_value[1]
    if j > 0: fill_arr[i,j-1,k]=fill_value[1]
        
    if k < fill_arr.shape[2]-1:fill_arr[i,j,k+1]=fill_value[2]
    if k > 0:fill_arr[i,j,k-1]=fill_value[2]
        
    return fill_arr


def second_nearest_neighbors(cells, fill_arr, fill_value):
    # checks the second nearest neighbor cells that need to be
    # checked in the next avalanche step
    for i in [cells[0]-2,cells[0]-1,cells[0]+1,cells[0]+2]:
        for j in [cells[1]-2,cells[1]-1,cells[1]+1,cells[1]+2]:
            for k in [cells[2]-2,cells[2]-1,cells[2]+1,cells[1]+2]:
                try:
                    fill_arr[cells[0],j,k]=fill_value[0]
                except:
                    pass
                try:
                    fill_arr[i,cells[1],k]=fill_value[1]
                except:
                    pass
                try:
                    fill_arr[i,j,cells[2]]=fill_value[2]
                except:
                    pass
    return fill_arr


def gradient(i,j,k,ng,f,df):
    # Find the gradient with the nearest neighbors
    # F = 0 outside of the region
    
    for x in range(len(df)):
        df[x] = f[i,j,k,x] + 0.0
    
        if i < (f.shape[0]-1): df[x] -= f[i+1,j,k,x]/6.0
        if i >= 0: df[x] -= f[i-1,j,k,x]/6.0
        
        if j < (f.shape[0]-1): df[x] -= f[i,j+1,k,x]/6.0
        if j >= 0: df[x] -= f[i,j-1,k,x]/6.0
            
        if k < (f.shape[0]-1): df[x] -= f[i,j,k+1,x]/6.0
        if k >= 0: df[x] -= f[i,j,k-1,x]/6.0
            
    dsqrt = np.sqrt(df[0]**2 + df[1]**2 + df[2]**2)
    return dsqrt


def distance(i,j,k,ng,f,dfm,fc):
    # redistributes cell across the neighbors
    con = fc / dfm
    
    for x in range(len(df)):
        # redistribute cell, not across boundary
        f[i,j,k,x] = f[i,j,k,x]- (6.0/7.0) * df[x] * con
        
        if i+1 < ng-1:
            f[i+1,j,k,x] = f[i+1,j,k,x] + (1.0/7.0) * df[x] * con
        if i-1 >= 0:
            f[i-1,j,k,x] = f[i-1,j,k,x] + (1.0/7.0) * df[x] * con
        
        if j+1 < ng-1:
            f[i,j+1,k,x] = f[i,j+1,k,x] + (1.0/7.0) * df[x] * con
        if j-1 >= 0:
            f[i,j-1,k,x] = f[i,j-1,k,x] + (1.0/7.0) * df[x] * con
            
        if k+1 < ng-1:
            f[i,j,k+1,x] = f[i,j,k+1,x] + (1.0/7.0) * df[x] * con
        if k-1 >= 0:
            f[i,j,k-1,x] = f[i,j,k-1,x] + (1.0/7.0) * df[x] * con
            
    return f



def add_fluctuations(f, fc, nloops, nfluc, random_seed):

    np.random.seed(random_seed)
    
    for i in range(nloops):
        print('nloops = ', i)
        
        nfl1 = np.zeros((f.shape[0],f.shape[0],f.shape[0]))
        nfl2 = np.zeros((f.shape[0],f.shape[0],f.shape[0]))
        df = np.zeros(f.shape[-1])
            
        nbin_e = np.zeros(f.shape[0]*f.shape[0]*f.shape[0])
        nbin_t = np.zeros(f.shape[0]*f.shape[0]*f.shape[0])
        nbin_p = np.zeros(f.shape[0]*f.shape[0]*f.shape[0])
        
        mev = 0 # count number of events during avalanche to determine size E
        nts = 0 # count number of steps in avalanche to determine duration T
        npk = 0 # keep track of peak number of unstable cells during cascade to determine P
        ninst=0 # reset count number of unstable cells in the time step
        
        for n in range(nfluc):
        
            cell = np.random.randint(0,f.shape[0],3) # pick random cell within the grid
            lower, upper = cell-1, cell+1
            pert = np.random.uniform(-0.03, 0.1, 3) # add perturbation between -0.03 and +0.1 fc

            # add perturbations to the grid
            for j in range(3):
                f[cell[0],cell[1],cell[2],j] = pert[j]*fc ## THIS MAY BE += ??

            # track neighbors of the perturbed cell
            nfl1 = check_neighbors(cell[0], cell[1], cell[2], nfl1, np.ones(3))
            dfm  = gradient(cell[0], cell[1], cell[2], f.shape[0], f, df)
            print(dfm)
            if dfm > fc:
                mev += 1
                ninst += 1

                f = distance(cell[0], cell[1], cell[2], f.shape[0], f, dfm, fc)

            # tracks avalanches in the 2nd nearest neighbors
            nfl2 = second_nearest_neighbors(cells, nfl2, np.ones(3))
            nfl1 = np.zeros(nfl1.shape)
            
            # if any cells are unstable, take next step in avalanche
            if ninst > 0:
                npk += 1
                nts += 1
                nfl1 = np.copy(nfl2)
                nfl2 = np.zeros(nfl1.shape)

            if mev>0:
                nbin_e[mev] += 1
                nbin_t[nts] += 1
                nbin_p[npk] += 1
    
        if i == 0:
            tab = Table()
            
        tab.add_column(Column(np.log10(nbin_e), 'E_{0:02d}'.format(i)))
        tab.add_column(Column(np.log10(nbin_t), 'T_{0:02d}'.format(i)))
        tab.add_column(Column(np.log10(nbin_p), 'P_{0:02d}'.format(i)))
    
    return tab


def main():
    ncells = 10
    nloops = 2
    nfluc  = 100#000000
    fc     = 7.0
    random_seed = 24
    
    f = np.zeros((ncells,ncells,ncells,3))
    df = np.zeros(3)
    f[:,:,:,0] = fc + 0.0
    
    tab = add_fluctuations(f, fc, nloops, nfluc, random_seed)
    tab.write('output.dat', format='ascii')

main()
