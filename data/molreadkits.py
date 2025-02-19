# sys.path.append('/vepfs/fs_users/ycjin/Delta-ML-Framework/dmf_tools/descriptior/unimol_tools')
from .coords2unimol import Coords2Unimol
from utils import pad_1d_tokens, pad_2d, pad_coords
import torch

class Mol2Input(object):
    def __init__(self):
        self.desc = ['unimol','Unimol']
        self.finetune = True
        self.input_list = []
        self.return_dict = {}

        if ('unimol' in self.desc) or ('Unimol' in self.desc):
            self.__init_unimol_func__()

        if 'new_desc' in self.desc:
            pass

    def __init_unimol_func__(self,):
        self.coords2unimol_input_func = Coords2Unimol()
        self.input_list.append('unimol')

    def mol2_inputs(self,mols):

        if 'unimol' in self.input_list:
            self.return_dict['unimol'] = self.mol2unimol_inputs(mols)

        return self.return_dict
    
    def coord2unimol_inputs_(self,coords,atoms):
        unimol_input = self.coords2unimol_input_func.get_tensors(atoms[0], coords)
        return unimol_input
    def coord2unimol_inputs(self,coords,atoms):
        unimol_inputs = {}
        if isinstance(coords,list):bs = len(coords)
        else:bs = coords.shape[0]
        # t1 = time.time()
        for i in range(bs):
            
            unimol_input = self.coords2unimol_input_func.get_tensor(atoms[i], coords[i])

            if unimol_inputs == {}:
                for k in unimol_input.keys():
                    if k == 'src_coord':
                        unimol_inputs[k] = [torch.tensor(unimol_input[k]).float()]
                    elif k == 'src_edge_type':
                        unimol_inputs[k] = [torch.tensor(unimol_input[k]).long()]
                    elif k == 'src_distance':
                        unimol_inputs[k] = [torch.tensor(unimol_input[k]).float()]
                    elif k == 'src_tokens':
                        unimol_inputs[k] = [torch.tensor(unimol_input[k]).long()]
            else:
                for k in unimol_input.keys():
                    if k == 'src_coord':
                        unimol_inputs[k].append(torch.tensor(unimol_input[k]).float())
                    elif k == 'src_edge_type':
                        unimol_inputs[k].append(torch.tensor(unimol_input[k]).long())
                    elif k == 'src_distance':
                        unimol_inputs[k].append(torch.tensor(unimol_input[k]).float())
                    elif k == 'src_tokens':
                        unimol_inputs[k].append(torch.tensor(unimol_input[k]).long())
        # t2 = time.time()

        for k in unimol_inputs.keys():
            if k == 'src_coord':
                unimol_inputs[k] = pad_coords(unimol_inputs[k], pad_idx=0.0)
            elif k == 'src_edge_type':
                unimol_inputs[k] = pad_2d(unimol_inputs[k], pad_idx=0)
            elif k == 'src_distance':
                unimol_inputs[k] = pad_2d(unimol_inputs[k], pad_idx=0.0)
            elif k == 'src_tokens':
                unimol_inputs[k] = pad_1d_tokens(unimol_inputs[k], pad_idx=0)
        # t3 = time.time()

        return unimol_inputs


    def mol2unimol_inputs(self,mols):
        if False:
            unimol_inputs = []
            for mol in mols:
                atoms = mol.get_chemical_symbols()
                coordinates = mol.get_positions()
                unimol_input = self.coords2unimol_input_func.get(atoms, coordinates)

                for k in unimol_input.keys():
                    if k == 'src_coord':
                        unimol_input[k] = torch.tensor([unimol_input[k]]).float()
                    elif k == 'src_edge_type':
                        unimol_input[k] = torch.tensor([unimol_input[k]]).long()
                    elif k == 'src_distance':
                        unimol_input[k] = torch.tensor([unimol_input[k]]).float()
                    elif k == 'src_tokens':
                        unimol_input[k] = torch.tensor([unimol_input[k]]).long()

                unimol_inputs.append(unimol_input)
        else:
            unimol_inputs = {}
            for mol in mols:
                atoms = mol.get_chemical_symbols()
                coordinates = mol.get_positions()
                unimol_input = self.coords2unimol_input_func.get(atoms, coordinates)
             
                if unimol_inputs == {}:
                    for k in unimol_input.keys():
                        if k == 'src_coord':
                            unimol_inputs[k] = [torch.tensor(unimol_input[k]).float()]
                        elif k == 'src_edge_type':
                            unimol_inputs[k] = [torch.tensor(unimol_input[k]).long()]
                        elif k == 'src_distance':
                            unimol_inputs[k] = [torch.tensor(unimol_input[k]).float()]
                        elif k == 'src_tokens':
                            unimol_inputs[k] = [torch.tensor(unimol_input[k]).long()]
                else:
                    for k in unimol_input.keys():
                        if k == 'src_coord':
                            unimol_inputs[k].append(torch.tensor(unimol_input[k]).float())
                        elif k == 'src_edge_type':
                            unimol_inputs[k].append(torch.tensor(unimol_input[k]).long())
                        elif k == 'src_distance':
                            unimol_inputs[k].append(torch.tensor(unimol_input[k]).float())
                        elif k == 'src_tokens':
                            unimol_inputs[k].append(torch.tensor(unimol_input[k]).long())

            for k in unimol_inputs.keys():
                if k == 'src_coord':
                    unimol_inputs[k] = pad_coords(unimol_inputs[k], pad_idx=0.0)
                elif k == 'src_edge_type':
                    unimol_inputs[k] = pad_2d(unimol_inputs[k], pad_idx=0)
                elif k == 'src_distance':
                    unimol_inputs[k] = pad_2d(unimol_inputs[k], pad_idx=0.0)
                elif k == 'src_tokens':
                    unimol_inputs[k] = pad_1d_tokens(unimol_inputs[k], pad_idx=0)

        return unimol_inputs
