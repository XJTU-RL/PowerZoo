# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import os
from pathlib import Path
from fnmatch import fnmatch

class LoadProfile:
    def __init__(self, steps, dss_folder_path, dss_file, worker_idx=None):
        self.steps = steps
    
        self.dss_folder_path = dss_folder_path
        self.loadshape_path = os.path.join(dss_folder_path, 'loadshape')
        if worker_idx is None:
            self.loadshape_dss = 'loadshape.dss'
        else:
            self.loadshape_dss = 'loadshape_' + str(worker_idx) + '.dss'
            
        if worker_idx is None:
            self.irrad_dss= 'irrad_up_down.dss'
        else:
            self.irrad_dss= 'irrad_up_down_'+ str(worker_idx) + '.dss'
        self.LOAD_NAMES = self.find_load_names(dss_file)
    
        self.FILES = []
        for f in os.listdir(self.loadshape_path):
            low = f.lower()
            if ('loadshape' in low) and low.endswith('.csv'):
                self.FILES.append(os.path.join(self.loadshape_path, f))
        

    def create_file_with_duty(self, fname):
        '''
            Create a new file named fname[:-4]_duty.dss
            In the new file,
            if any load is created, then the load is associated with its duty loadshape.
        '''
        fin = open(os.path.join(self.dss_folder_path, fname), 'r')
        fout = open(os.path.join(self.dss_folder_path, fname[:-4] + '_duty.dss'), 'w')
        for line in fin:
            if not line.lower().startswith('new load.') or ('duty' in line):
                fout.write(line)
            else:
                line = line.strip()
                if '!' in line: line = line[:line.find('!')].strip() # remove inline comment
                if '//' in line: line = line[:line.find('//')].strip() # remove inline comment
                spt = list(filter(None, line.split(' '))) # filter out the empty string
                load = spt[1].split('.',1)[1]
                fout.write(line + ' duty=loadshape_' + load + '\n')
        fin.close()
        fout.close()
   
    def add_redirect_and_mode_at_main_duty_dss(self, main_duty_dss):
        '''
            Add redirect loadshape (& load file if any) 
            and set duty mode at the main duty dss file

        Args:
            main_duty_dss: the file name of the main duty dss file
        
        Returns:
            the load dss file (if any) associated with the main dss file
        '''
        # load the file
        fin = open(os.path.join(self.dss_folder_path, main_duty_dss), 'r')
        lines = [line for line in fin]
        fin.close()

        # overwrite the file
        found_load, redirect_load = False, False
        load_file = None
        fout = open(os.path.join(self.dss_folder_path, main_duty_dss), 'w')
        for line in lines:
            low = line.strip().lower()
            if '!' in low: low = low[:low.find('!')].strip() # remove inline comment
            if '//' in low: low = low[:low.find('//')].strip() # remove inline comment
            if (not found_load) and 'load' in low and not low.startswith('~'):
                fout.write('! add loadshape\n')
                fout.write('redirect ' + self.loadshape_dss + '\n\n')
                found_load = True

            low = low[:-4] if len(low)>=4 else ''
            if (not redirect_load) and low.startswith('redirect'):
                if low.endswith('loads') or low.endswith('load'):
                    load_file = list(filter(None, line.strip().split(' ')))[1] # remove the empty string
                    fout.write('redirect ' + load_file[:-4] + '_duty.dss\n')
                    redirect_load = True
                elif low.endswith('loads_duty') or low.endswith('load_duty'):
                    load_file = list(filter(None, line.strip().split(' ')))[1] # remove the empty string
                    fout.write(line)
                    redirect_load = True
                else: fout.write(line)
            else: fout.write(line)
        
        assert found_load, 'cannot find load at ' + main_duty_dss

        fout.write('Set mode=Daily number=1 hour=0 stepsize=3600 sec=0\n')
        fout.close()
        
        return load_file

    def find_load_file_from(self, main_dss):
        fin = open(os.path.join(self.dss_folder_path, main_dss), 'r')
        load_file = None
        for line in fin:
            low = line.strip().lower()
            if '!' in low: low = low[:low.find('!')].strip() # remove inline comment
            if '//' in low: low = low[:low.find('//')].strip() # remove inline comment
            low = low[:-4] if len(low)>=4 else ''
            if low.startswith('redirect') and \
               (low.endswith('loads') or low.endswith('load') \
                or low.endswith('loads_duty') or low.endswith('load_duty') ):
                
                load_file = list(filter(None, line.strip().split(' ')))[1]
                break
        return load_file

    def find_load_names(self, main_dss):
        '''
            Find the loads with duty loadshapes at main dss or the load dss files.
            If there is none, 
            then generate new files (annotated _duty) with duty loadshapes.
        '''
        def find_load_name(fname, names):
            file_path = os.path.join(self.dss_folder_path, fname)
            assert os.path.exists(file_path), file_path + ' not found'
            
            needs_load_duty, duty_mode = False, False
            with open(file_path, 'r') as fin:
                for line in fin:
                    low = line.strip().lower()
                    if low.startswith('new load.'):
                        if 'duty' in low:
                            spt = line.split(' ')
                            spt = list(filter(None, spt)) # filter out the empty string
                            names.append(spt[1].split('.',1)[1])
                        else: needs_load_duty = True
                    if low.startswith('set mode=duty'):
                        duty_mode = True
            return needs_load_duty, duty_mode
        names = []

        # add from the main dss file
        needs_load_duty, duty_mode = find_load_name(main_dss, names)
        if needs_load_duty or (not duty_mode):
            ## Create a new _duty file. Add duty loadshape if needed
            self.create_file_with_duty(main_dss) #todo,多线程创建dss文件在这里

            ## add redirect and set duty mode at the new _duty file
            load_file = self.add_redirect_and_mode_at_main_duty_dss(\
                                              main_dss[:-4]+'_duty.dss')
        else:
            load_file = self.find_load_file_from(main_dss)

        # add from the other load files
        if load_file is not None:
            if 'duty' in load_file:
                needs_load_duty, _ = find_load_name(load_file, names)
                assert (not needs_load_duty), 'invalid content in ' + load_file
            else:
                needs_load_duty, _ = find_load_name(load_file, names)
                if needs_load_duty: self.create_file_with_duty(load_file)

        # check empty or duplicate load
        assert len(names)>0, 'daliy load not found. Consider modifying from the auto-generated file annotated with _duty'
        assert len(names) == len(set(names)), 'duplicate load names'
        
        return names     

    def gen_loadprofile(self, scale=1.0):
        
        try:
            dfs = []
            for f in self.FILES:
                dfs.append(pd.read_csv(f, header=None))
            assert len(dfs) > 0, "loadshape 目录下没有负荷曲线 CSV 文件"
            df = pd.concat(dfs).rename(columns={0: 'mul'}).reset_index(drop=True)
            if scale != 1.0:
                df['mul'] = df['mul'] * scale
        except Exception as e:
            # P0-8：原 bare except 静默吞错，随后 len(df) 必然 NameError；改为带上下文直接抛出
            raise RuntimeError(
                f"生成负荷曲线失败：{e}（请检查 {self.loadshape_path} 下的负荷 CSV 文件）"
            ) from e
        
        # Since we've concatenated all loadshape file together,
        # the first load uses the top steps*days data points
        # the second load uses the second chunk and so on.
        
        # compute totoal number of episodes
        episodes = len(df) // ( self.steps * len(self.LOAD_NAMES) )
        
        # stop here only if the loadprofile folders exist
        checks = [fnmatch(f, '0*') for f in os.listdir(self.loadshape_path)]
        scale_txt = os.path.join(self.loadshape_path, 'scale.txt')
        fscale = np.genfromtxt(scale_txt).reshape(1)[0] if os.path.exists(scale_txt) else None
        if sum(checks)==episodes and fscale==scale:
            return episodes

        # save the scale for future use
        np.savetxt(scale_txt, np.array([scale]))

        # insert loadname, day, step columns
        load_col, episode_col, step_col = [], [], []
        for i in range( self.steps*episodes*len(self.LOAD_NAMES) ):
            load_col.append(self.LOAD_NAMES[i//(self.steps*episodes)])
            episode_col.append((i//self.steps)%episodes)
            step_col.append(i%self.steps)
        df = df[:len(load_col)]
        df['load'] = load_col
        df['episode'] = episode_col
        df['step'] = step_col
        
        # sort and output
        df = df.sort_values(by=['episode','load','step'])[['episode','load','step','mul']].reset_index(drop=True)
        for episode in range(episodes):
            if not os.path.exists(os.path.join(self.loadshape_path, str(episode).zfill(3))):
                os.mkdir(os.path.join(self.loadshape_path, str(episode).zfill(3)))
            sdf = df[df['episode']==episode]
            for load in self.LOAD_NAMES:
                series = sdf[sdf['load']==load]['mul']
                series.to_csv(\
                  os.path.join(self.loadshape_path, str(episode).zfill(3), load+'.csv'),\
                  header=False, index=False)

        return episodes # number of distinct epochs
    
    def choose_loadprofile(self, idx):
        assert os.path.exists(os.path.join(self.loadshape_path, str(idx).zfill(3))), 'idx does not exist'
        
        with open(os.path.join(self.dss_folder_path, self.loadshape_dss), 'w') as fp:
            for load_name in self.LOAD_NAMES:
                fp.write(f'New Loadshape.loadshape_{load_name} npts={self.steps} sinterval={1} ' +
                    'mult=(file=./' + os.path.join('loadshape', str(idx).zfill(3), load_name+'.csv') + ')\n' )

        return os.path.join(self.loadshape_path, str(idx).zfill(3))
    
    def choose_irrad_profile(self, style):
        assert os.path.exists(os.path.join(self.loadshape_path,style+'.csv')), 'idx does not exist'
        
        with open(os.path.join(self.dss_folder_path, self.irrad_dss), 'w') as fp:
            fp.write(f'New Loadshape.MyIrrad npts={self.steps} sinterval={1} ' +
            'mult=(file=./' + os.path.join('loadshape', style+'.csv') + ')\n' )

        return os.path.join(self.loadshape_path)

    def get_loadprofile(self, idx):
        folder_path  = os.path.join(self.loadshape_path, str(idx).zfill(3))
        csv_paths = os.listdir(folder_path)
        temp_loads = []
        for csv in csv_paths:
            csv_file = os.path.join(folder_path,csv)
            load = pd.read_csv(csv_file, header=None, names=[csv.split('.')[0]])
            temp_loads.append(load)

        all_loads = pd.concat(temp_loads, axis=1)
        return all_loads
