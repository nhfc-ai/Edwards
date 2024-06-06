
import pandas as  pd
from datetime import datetime as dt
import numpy as np
import pickle
from .standard_config import * 



def dump_data_into_text(in_file, out_file):
    df = pd.read_csv(in_file, index_col=False).fillna('')
    cycle_ids = set(df['artcycle_id'])
    final_data = []
    while len(cycle_ids) > 0:
        _cid = cycle_ids.pop()
        _text, _demo_text, _diag, _er_label, _final_label = '', '', '', '', ''
        _upper_level_temp_list, _upper_level_temp_diag = [], []
        for i, _visit_records in enumerate(df.loc[df['artcycle_id'] == _cid].itertuples()):
            _cycle_day, _position, _e2, _fsh, _lh, _p4, _hcg, _follicles, _cyst, _endo, _age, \
                _cycle_len, _bmi, _diagnosis, _final_results, _folli, _orli, _clomid, _fem, _ocp, \
                _est, _lupron, _ovidrel, _er_results = _visit_records[5:]

            if _final_results != '':
                _final_label = _final_results
            
            if _er_results != '':
                _er_label = _er_results
            
            if len(''.join(str(i) for i in _visit_records[7:15])) + len(''.join(str(i) for i in _visit_records[20:28])) == 0:
                continue

            if i == 0:
                _demo_text = ', '.join(str(item) for item in [_age, _cycle_len, _bmi])
            
            if _cycle_day != '':
                _cycle_day = encode_cycle_day(_cycle_day)
            
            _temp_list = []
            for i in [_cycle_day, _e2, _fsh, _lh, _p4, _hcg, _follicles, _cyst, _endo, convert_rare_dose(_folli), 
                    convert_rare_dose(_orli), convert_rare_dose(_clomid), convert_rare_dose(_fem), convert_rare_dose(_ocp),
                    convert_rare_dose(_est), _lupron, _ovidrel]:
                if i != '':
                    _temp_list.append(i)
            if len(_temp_list) > 0:
                _temp_list = ['[CLS]'] + _temp_list + ['[SEP]']
                _upper_level_temp_list.append(', '.join(str(item) for item in _temp_list))
                _upper_level_temp_diag.append(_diagnosis)
        assert len(_upper_level_temp_list) == len(_upper_level_temp_diag)

        _text = ', '.join(str(item) for item in _upper_level_temp_list)
        _diag = ', '.join(str(item) for item in _upper_level_temp_diag)

        final_data.append([_cid, _text, _demo_text, _diag, _er_label, _final_label])
    pd.DataFrame(final_data, columns=DATASET_COLUMN).to_csv(out_file)
    
def convert_rare_dose(dose):
    if dose in RARE_DOSE_CONVERTER:
        return RARE_DOSE_CONVERTER[dose]
    return dose


def split_ori_dataset_to_testset(ori_file, start_date, out_file):
    assert type(start_date).__name__ == 'datetime'
    ori_df = pd.read_csv(ori_file, index_col=False).fillna('')[DATASET_COLUMN]
    out_list = []
    for row in ori_df.itertuples():
        try:
            _, _cid, _d, _core_string, _demo_string, _diag_string, _er_result, _final_result = row
            if dt.strptime(_d, '%Y-%m-%d') >= start_date:
                _core_list = _core_string.split(', ')
                end_token_idx = np.where(np.array(_core_list)=='[SEP]')[0]
                for i, _idx in enumerate(end_token_idx):
                    _new_cid = _cid + " " + str(i)
                    out_list.append([_new_cid, _d, ', '.join(_core_list[:_idx+1]), _demo_string, _diag_string, _er_result, _final_result])
                
            else:
                _new_cid = _cid + " 0"
                out_list.append([_new_cid, _d, _core_string, _demo_string, _diag_string, _er_result, _final_result])
        except Exception as e:
            print(row)
            print(e)
            input()
    pd.DataFrame(out_list, columns=DATASET_COLUMN).to_csv(out_file)

