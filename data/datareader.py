import pandas as pd
import numpy as np
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem
from utils import LMDBDataset
import pickle
import lmdb
import os

def prepare_data(file_path,save_path = None, _col_name='smiles',test_size = 0.2):
    print('Preparing data...')
    data_path = smiles_to_xyz(file_path,save_path,_col_name)
    return data_to_lmdb(data_path,test_size = test_size)

def smiles_to_xyz(file_path, save_path = None, _col_name='smiles'):
    print('Converting SMILES to XYZ...')
    if save_path is None:
        save_path = os.path.splitext(file_path)[0]

    data_save_path = save_path + '/data/'
    if os.path.exists(data_save_path + 'data.csv'):
        print('Data already exists, skipping...',)
        return data_save_path

    df = pd.read_csv(file_path)

    try:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass
    
    data_dict = {
        'file':[],
    }

    for i in df.columns:
        if i != _col_name:
            data_dict[i] = []
    
    xyz_save_path = save_path + '/xyz/'
    if not os.path.exists(xyz_save_path):
        os.mkdir(xyz_save_path)
    id = 0
    for i in range(len(df[_col_name])):
        smiles = df[_col_name][i]
        mol = Chem.MolFromSmiles(smiles)
        hs = Chem.AddHs(mol) 
        try:
            AllChem.EmbedMultipleConfs(hs, useExpTorsionAnglePrefs=True, useRandomCoords=True, useBasicKnowledge=True, numConfs=1)
            AllChem.MMFFOptimizeMolecule(hs, mmffVariant='MMFF94s', maxIters=1000, ignoreInterfragInteractions=False)
            Chem.MolToXYZFile(hs,xyz_save_path + f'{id}.xyz')
            data_dict['file'].append(xyz_save_path + f'{id}.xyz')
            for col in df.columns:
                if col != _col_name:
                    data_dict[col].append(float(df[col][i]))
            id+=1
        except:
            print(smiles)
            continue
    data=pd.DataFrame(data_dict)
    data_save_path = save_path + '/data/'
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)
    data.to_csv(data_save_path + 'data.csv')

    return data_save_path

def data_to_lmdb(data_path,test_size = 0.2):
    print('Converting data to LMDB...')
    data = pd.read_csv(data_path + 'data.csv')
    try:
        data.drop(columns=['Unnamed: 0'], inplace=True)
    except:
        pass

    from sklearn.model_selection import train_test_split
    x = np.array(data['file'])
    data.drop(columns=['file'])
    X_train, X_test, y_train, y_test = train_test_split(x, np.array(data.drop(columns=['file'])), test_size=test_size, random_state=42)
    train_lmdbdataset_name = data_path +'train_data.lmdb'
    env_new = lmdb.open(
            train_lmdbdataset_name,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )

    txn_write = env_new.begin(write=True)
    for idx in range(len(X_train)):
        item = {}
        atom = read(X_train[idx])
        atype = np.array(atom.get_chemical_symbols())
        item['coord'] = atom.get_positions()[atype != 'H']
        item['atype'] = atype[atype != 'H']
        item['target'] = np.array(y_train[idx])
        txn_write.put(f"{idx}".encode("ascii"), pickle.dumps(item, protocol=-1))
    txn_write.commit()
    env_new.close()

    val_lmdbdataset_name = data_path +'val_data.lmdb'
    env_new = lmdb.open(
            val_lmdbdataset_name,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )

    txn_write = env_new.begin(write=True)
    for idx in range(len(X_test)):
        item = {}
        atom = read(X_test[idx])
        atype = np.array(atom.get_chemical_symbols())
        item['coord'] = atom.get_positions()[atype != 'H']
        item['atype'] = atype[atype != 'H']
        item['target'] = np.array(y_test[idx])
        txn_write.put(f"{idx}".encode("ascii"), pickle.dumps(item, protocol=-1))
    txn_write.commit()
    env_new.close()
    print('Done')
    return train_lmdbdataset_name,val_lmdbdataset_name

def data_to_lmdb_from_npy(coord,atype,label,test_size = None,save_path = './example/temp/'):
    print('Converting data to LMDB...')
    if test_size is None:
        train_lmdbdataset_name = save_path + 'train_data.lmdb'
        env_new = lmdb.open(
                train_lmdbdataset_name,
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(100e9),
            )

        txn_write = env_new.begin(write=True)
        for idx in range(len(coord)):
            item = {}
            item['coord'] = coord[idx]
            item['atype'] = np.array(atype)
            item['target'] = np.array(label[idx])
            txn_write.put(f"{idx}".encode("ascii"), pickle.dumps(item, protocol=-1))
        txn_write.commit()
        env_new.close()
        print('Done')
        return train_lmdbdataset_name