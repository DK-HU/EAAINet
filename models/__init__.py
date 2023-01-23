from .multi_scale_trans import Multi_Scale_Trans

def get_model(name, dataset):
    return {
            'multi_scale_trans': Multi_Scale_Trans()
           }[name]
