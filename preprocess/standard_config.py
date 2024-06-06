


MONITOR_ITEM_LENGTH = 15

TOKEN_COLUMNS = ['cycle_date', 'e2', 'lh', 'p4', 'bhcg', 'afs', 'folli', 'endo', 'orli', 'clomid', 'fem', 'ocp', 'est', 'trigger']

LABEL_COLUMNS = ['next_'+i for i in TOKEN_COLUMNS]


BASELINE_TRAIN_LABELS = ['fsh', 'e2','lh', 'p4', 'endo', 'cyst', 'folli', 'clomid', 'fem', 'orli','ocp', 'trigger', 'cycleday', 'afs']

BASELINE_TEST_LABELS = ['fsh', 'e2','lh', 'p4', 'endo', 'cyst', 'folli', 'clomid', 'fem', 'orli','ocp', 'trigger', 'cycleday', 'sono']

DEMO_COLUMNS = ['age', 'cycle_length', 'bmi']

CID_COLUMNS = ['cid']

DATASET_COLUMN = ['artcycle_id', 'date', 'core_text', 'demo_text', 'diag', 'er_result', 'final_result', 'mii_label', 'twopn_label', 'blst_label']

BASELINE_LABEL_DICT = {'cycle day: 0-3': 0, 'cycle day: 4-7': 1, 
                    'cycle day: 8-11': 2, 'cycle day: 12-15': 3, 
                    'cycle day: 16-19': 4, 'cycle day: 20-23': 5, 
                    'cycle day: 24-27': 6, 'cycle day: 28-': 7, 
                    'fsh: 0-5': 0, 'fsh: 5-15': 1, 'fsh: 15-30': 2, 
                    'fsh: 30-40': 3, 'fsh: 40-': 4, 'e2: 0-50': 0,
                    'e2: 50-100': 1, 'e2: 100-200': 2, 'e2: 200-500': 3,
                    'e2: 500-1000': 4, 'e2: 1000-1500': 5, 'e2: 1500-2000': 6,
                    'e2: 2000-3000': 7, 'e2: 3000-': 8, 'lh: 0-1': 0,
                    'lh: 1-1.5': 1, 'lh: 1.5-2': 2, 'lh: 2-3': 3,
                    'lh: 3-5': 4, 'lh: 5-7': 5, 'lh: 7-10': 6, 'lh: 10-': 7,
                    'p4: 0-0.5': 0, 'p4: 0.5-1': 1, 'p4: 1-1.5': 2, 'p4: 1.5-2': 3,
                    'p4: 2-': 4, 'bhcg: 0-5': 0, 'bhcg: 5-': 1, 'sono: quiet': 0, 
                    'sono: ovulate': 1, 'sono: >20: >15% and >15: >35%': 2, 
                    'sono: >20: >15% and >15: <35%': 3, 'sono: >20: 0-15% and >15: >35%': 4, 
                    'sono: >20: 0-15% and >15: <35%': 5, 'sono: >15: >35% and >8: >35%': 6, 
                    'sono: >15: <35% and >8: >35%': 7, 'afs: 1-5': 0, 'afs: 6-10': 1, 
                    'afs: 10-20': 2, 'afs: 20+': 3,'endo: <5': 0, 
                    'endo: 5-7': 1, 'endo: 7-13': 2, 'endo: 13-': 3, 'folli: 75.0': 0, 'folli: 50.0': 0, 'folli: 65.0': 0, 'folli: 81.0': 0, 'folli: 87.5': 0,
                    'folli: 75.0 eod': 1, 'folli: 50.0 eod': 1, 'folli: 100.0 eod': 1, 'folli: 150.0': 2, 'folli: 125.0': 2, 'folli: 175.0': 2, 'folli: 112.5': 2,
                    'folli partially missed': 2, 'folli: 150.0 eod': 3, 'folli: 125.0 eod': 3, 'folli: 175.0 eod': 3,
                    'folli: 225.0': 4, 'folli: 250.0': 4, 'folli: 220.0': 4, 'folli: 275.0': 4, 'folli: 200.0': 4, 'folli: 225.0 eod': 5, 'folli: 255.0 eod': 5, 'folli: 220.0 eod': 5, 'folli: 200.0 eod': 5, 'folli: no dose' : 6, 
                  'clomid: 50.0': 0, 'clomid: 13.0': 0, 'clomid: 20.0': 0, 'clomid: 28.0': 0, 'clomid: 32.0': 0, 'clomid: 34.0': 0, 'clomid: 40.0': 0,'clomid: 12.5 eod': 0, 'clomid: 22.0': 0, 'clomid: 30.0': 0, 'clomid: 36.0': 0, 'clomid: 37.5': 0, 'clomid: 35.0': 0, 'clomid: 50.0 eod': 0, 'clomid: 75.0 eod': 0, 'clomid partially missed': 0, 'clomid: 75.0': 1, 'clomid: 81.0': 1, 'clomid: 80.0': 1, 'clomid: 100.0 eod': 1, 'clomid: 100.0': 1, 'clomid: 150.0 eod': 1, 'clomid: 200.0': 1, 'clomid: no dose' :2, 
                  'fem: 2.5': 0, 'fem: 2.667': 0, 'fem: 1.333': 0, 'fem: 2.0': 0, 'fem: 3.0': 0, 'fem: 1.6': 0, 'fem: 1.5': 0, 'fem: 2.35': 0, 'fem: 1.8': 0, 'fem: 1.75': 0, 'fem: 1.25 eod': 0, 'fem: 1.8 eod': 0, 'fem: 2.5 eod': 0, 'fem: 3.0 eod': 0, 'fem: 5.0 eod': 0, 'fem: 3.333': 0, 'fem: 3.5': 0, 'fem: 3.7': 0, 'fem: 3.667': 0, 'fem: 3.75': 0, 'fem partially missed': 0, 'fem: 4.5': 1, 'fem: 4.0': 1, 'fem: 5.0': 1, 'fem: no dose': 2, 'orli partially missed': 0, 
                  'orli: 0.125': 0, 'orli: 0.133': 0, 'orli: 0.172': 0, 'orli: 0.182': 0, 'orli: 0.185': 0, 'orli: 0.154': 0,
                  'orli: 0.167': 0, 'orli: 0.15': 0, 'orli: 0.143': 0, 'orli: 0.136': 0, 'orli: 0.125 eod': 1, 'orli: 0.25': 2, 'orli: 0.2': 2,
                  'orli: 0.259': 2, 'orli: 0.28': 2, 'orli: 0.273': 2, 'orli: 0.235': 2, 'orli: 0.231': 2, 'orli: 0.207': 2, 
                  'orli: 0.211': 2, 'orli: 0.214': 2, 'orli: 0.25': 2, 'orli: 0.241': 2, 'oril: 0.25': 2,  'orli: 0.286': 2, 'orli: 0.263': 2, 'orli: 0.261': 2, 'orli: 0.292': 2, 'orli: 0.3': 2, 'orli: 0.296': 2,
                  'orli: 0.316': 2, 'orli: 0.312': 2,  'orli: 0.357': 2, 'orli: 0.364': 2, 'orli: 0.385': 2, 'orli: 0.4': 2, 
                  'orli: 0.367':2, 'orli: 0.35':2, 'orli: 0.269':2, 'orli: 0.32':2, 'orli: 0.417': 2, 
                  'orli: 0.308': 2, 'orli: 0.389': 2, 'orli: 0.455': 2, 'orli: 0.44': 2, 'orli: 0.462': 2, 'orli: 0.435': 2, 
                  'orli: 0.444': 2, 'orli: 0.429': 2, 'orli: 0.25 eod': 3, 'orli: 0.412 eod': 3, 'orli: 0.4 eod': 3,
                  'orli: no dose': 4, 'ocp: 0.5': 0, 'ocp: 0.583': 0, 'ocp: 0.533': 0, 'ocp: 0.545': 0, 'ocp: 0.526': 0, 'ocp: 0.571': 0, 
                  'ocp: 0.579': 0, 'ocp: 0.522': 0, 'ocp: 0.55': 0, 'ocp: 0.6': 0, 'ocp: 0.636': 0, 'ocp: 0.615': 0, 
                  'ocp: 0.667': 0, 'ocp: 0.643': 0, 'ocp: 0.69': 0, 'ocp: 0.727': 0, 'ocp: 0.857 eod': 0, 'ocp: 0.75': 0, 'ocp: 0.786': 0, 
                  'ocp: 0.733': 0, 'ocp: 0.5 eod': 1, 'ocp: 0.667 eod': 1, 'ocp: 0.538 eod': 1, 'ocp: 0.545 eod': 1, 
                  'ocp: 0.625 eod': 1, 'ocp partially missed':0, 'ocp: 1.0': 2, 'ocp: 0.909': 2, 'ocp: 0.875': 2, 'ocp: 0.9': 2, 
                  'ocp: 0.917': 2, 'ocp: 0.846': 2, 'ocp: 0.833': 2, 'ocp: 0.778': 2,
                  'ocp: 1.0 eod': 3, 'ocp: no dose': 4, 'est: 1.0': 0, 'est: 2.0': 1, 
                  'est: 4.0': 2, 'est: 6.0': 3, 'est: po2pv2': 4, 'est: no dose': 5,
                 'ovidrel': 0, 'lupron': 1, 'ovidrel+lupron': 2, 'no trigger': 3, 
                 'too early to trigger': 4}

DEFAULT_CYCLE_LENGTH = 28
FAKE_FOLLI_BOUND = 15
FAKE_FOLLI_DUPLICATE_BOUND = 10

DATASET_COLUMN = ['artcycle_id', 'core_text', 'demo_text', 'diag', 'er_result', 'final_result']

DATA_COLUMNS = ['cycle_id', 'patient_id', 'date', 'cycle_date', 'position', 'e2', 'fsh', 'lh', 'p4',
                 'hcg', 'follicles', 'cyst', 'endo', 'age', 'cycle_length', 'bmi', 'diagnosis', 'results',
                 'folli', 'orli', 'clomid', 'fem', 'ocp', 'est', 'lupron', 'ovidrel', 'er_result']

ER_CODE = ['Terminate', 'LPS']

TRIGGER_MED = ['Lupron', 'Ovidrel']

PROCEDURE_CODE = {'blood test': 0, 'sonography': 1, 'egg retrieval': 2, 
                'frozen transfer': 3, 'fresh transfer': 4, 'no procedure': 5}

DIAG_CODE = {'stim preparation': 400, 'conventional ivf': 401, 'mini ivf': 402, 'ultra ivf': 403, 'nivf': 404, 'continue stim': 405,
             'continue stim w/o med': 406, 'trigger': 407, 'follow the plan': 408, 'lps': 409, 'hrt fet prep': 410, 'nfet prep': 411, 
             'cont. hrt fet prep': 412, 'cont. nfet prep': 413, 'hrt fet phase 1': 414, 'hrt fet phase 2': 415, 'nfet phase 1': 416,
             'nfet phase 2': 417}


RESULT_CODE = {'ovulate': 100, 'no egg': 101, 'egg 1-5': 102, 'egg 5-10': 103,
             'egg 10-20': 104, 'egg 20-': 105, 'egg 1-5 with ovulation': 106, 'egg 5-10 with ovulation': 107,
             'egg 10-20 with ovulation': 108, 'egg 20- with ovulation': 109, 'BLST: no 2pn': 110,
              'BLST: <50% 2pn but no blst': 111, 'BLST: <50% 2pn and blst': 112,
               'BLST: >50% 2pn and blst': 113, 'BLST: >50% 2pn but no blst': 114, 'MII: no MII': 115, 'MII: <50% MII': 116,
               'MII: >50% MII': 117, 'no preg': 118, 'chemical preg': 119, 'sac': 120,
               'live': 121}

TEST_RESULT_CODE = {'fsh: 0-5': 200, 'fsh: 5-15': 201, 'fsh: 15-30': 202, 
                    'fsh: 30-40': 203, 'fsh: 40-': 204, 'e2: 0-50': 205,
                    'e2: 50-100': 206, 'e2: 100-200': 207, 'e2: 200-500': 208,
                    'e2: 500-1000': 209, 'e2: 1000-1500': 210, 'e2: 1500-2000': 211,
                    'e2: 2000-3000': 212, 'e2: 3000-': 213, 'lh: 0-1': 214,
                    'lh: 1-1.5': 215, 'lh: 1.5-2': 216, 'lh: 2-3': 217,
                    'lh: 3-5': 218, 'lh: 5-7': 219, 'lh: 7-10': 220, 'lh: 10-': 221,
                    'p4: 0-0.5': 222, 'p4: 0.5-1': 223, 'p4: 1-1.5': 224, 'p4: 1.5-2': 225,
                    'p4: 2-': 226, 'bhcg: 0-5': 227, 'bhcg: 5-': 228, 'sono: quiet': 229, 
                    'sono: ovulate': 230, 'sono: >20: >15% and >15: >35%': 231, 
                    'sono: >20: >15% and >15: <35%': 232, 'sono: >20: 0-15% and >15: >35%': 233, 
                    'sono: >20: 0-15% and >15: <35%': 234,
                    'sono: >15: >35% and >8: >35%': 235, 'sono: >15: <35% and >8: >35%': 236, 
                    'afs: 1-5': 237, 'afs: 6-10': 238, 'afs: 10-20': 239, 'afs: 20+': 240,'endo: <5': 241, 
                    'endo: 5-7': 242, 'endo: 7-13': 243, 'endo: 13-': 244, 'cyst: > 15': 245}

BLOOD_TEST_CODE = ['fsh: 0-5', 'fsh: 5-15', 'fsh: 15-30', 'fsh: 30-40', 'fsh: 40-', 'e2: 0-50', 
                    'e2: 50-100', 'e2: 100-200', 'e2: 200-500', 'e2: 500-1000', 'e2: 1000-1500', 
                    'e2: 1500-2000', 'e2: 2000-3000', 'e2: 3000-', 'lh: 0-1', 'lh: 1-1.5', 'lh: 1.5-2', 
                    'lh: 2-3', 'lh: 3-5', 'lh: 5-7', 'lh: 7-10', 'lh: 10-', 'p4: 0-0.5', 'p4: 0.5-1', 
                    'p4: 1-1.5', 'p4: 1.5-2', 'p4: 2-', 'bhcg: 0-5', 'bhcg: 5-', 'sono: quiet']

SONO_TEST_CODE = ['sono: ovulate', 'sono: >20: >15% and >15: >35%', 'sono: >20: >15% and >15: <35%',
                  'sono: >20: 0-15% and >15: >35%', 'sono: >20: 0-15% and >15: <35%', 
                  'sono: >15: >35% and >8: >35%', 'sono: >15: <35% and >8: >35%', 
                  'afs: 1-5', 'afs: 6-10', 'afs: 10-20', 'afs: 20+', 'endo: <5', 'endo: 5-7', 
                  'endo: 7-13', 'endo: 13-', 'cyst: > 15']

CYCLE_DAY_CODE_LIST = ['cycle day: 0-3', 'cycle day: 4-7', 'cycle day: 8-11', 'cycle day: 12-15', 'cycle day: 16-19', 
                    'cycle day: 20-23', 'cycle day: 24-27', 'cycle day: 28-']

TREATMENT_CODE = ['folli: 75.0', 'folli: 75.0 eod', 'folli: 150.0', 'folli: 150.0 eod', 'folli: 225.0', 'folli: 225.0 eod', 'folli: no dose', 
                  'clomid: 50.0', 'clomid: 100.0', 'clomid: no dose', 'fem: 2.5', 'fem: 5.0', 'fem: no dose', 'orli: 0.125', 
                  'orli: 0.125 eod', 'orli: 0.25', 'orli: 0.25 eod', 'orli: no dose', 'ocp: 0.5', 'ocp: 0.5 eod', 'ocp: 1.0', 
                  'ocp: 1.0 eod', 'ocp: no dose', 'est: 1.0', 'est: 2.0', 'est: 4.0', 'est: 6.0', 'est: po2pv2', 'est: no dose',
                 'ovidrel', 'lupron', 'ovidrel+lupron', 'no trigger', 'too early to trigger']

TREATMENT_CODE_INDEX = {'folli': list(range(51,59))+[94], 'clomid': list(range(63,66))+[95], 'fem': list(range(66,69))+[96],
                        'orli': list(range(69,74))+[97], 'ocp': list(range(74,79))+[98], 'est': list(range(79,85))+[99], 
                        'trigger':  list(range(90,94))+[101]}

NEXT_VISIT_INDEX = {'fsh': list(range(5,10)), 'e2': list(range(10,19)), 'lh': list(range(19,27)),
                    'p4': list(range(27,32)), 'bhcg': list(range(32,34)), 'endo': list(range(46,50)), 
                    'sono': list(range(34,46)), 'cycleday': list(range(102,110))}

CODE_SUB_GROUP = [list(range(5,10)), list(range(10,19)), list(range(19,27)), 
                    list(range(27,32)), list(range(32,34)), list(range(34,46)),
                    list(range(46,50)), list(range(50,51)), 
                    list(range(51,59))+[94], list(range(59,63)), list(range(63,66))+[95],
                    list(range(66,69))+[96], list(range(69,74))+[97], list(range(74,79))+[98],
                    list(range(79,85))+[99], list(range(85,89))+[100], list(range(89,90)),
                    list(range(90,94))+[101], list(range(102,110))]

CODE_SUB_GROUP_W_KEYS = {'fsh': list(range(5,10)), 'e2': list(range(10,19)), 'lh': list(range(19,27)), 
                    'p4': list(range(27,32)), 'bhcg': list(range(32,34)), 'sono': list(range(34,46)),
                    'endo': list(range(46,50)), 'cyst': list(range(50,51)), 
                    'folli': list(range(51,59))+[94], 'menopur': list(range(59,63)), 'clomid': list(range(63,66))+[95],
                    'fem': list(range(66,69))+[96], 'orli': list(range(69,74))+[97], 'ocp': list(range(74,79))+[98],
                    'est': list(range(79,85))+[99], 'prog': list(range(85,89))+[100], 'gani': list(range(89,90)),
                    'trigger': list(range(90,94))+[101], 'cycleday': list(range(102,110))}


CODE_SUB_GROUP_W_KEYS_FOR_BASELINE = {'fsh': list(range(5,10)), 'e2': list(range(10,19)), 'lh': list(range(19,27)), 
                    'p4': list(range(27,32)), 'bhcg': list(range(32,34)), 'afs': list(range(34,46)),
                    'endo': list(range(46,50)), 'cyst': list(range(50,51)), 
                    'folli': list(range(51,59))+[94], 'menopur': list(range(59,63)), 'clomid': list(range(63,66))+[95],
                    'fem': list(range(66,69))+[96], 'orli': list(range(69,74))+[97], 'ocp': list(range(74,79))+[98],
                    'est': list(range(79,85))+[99], 'prog': list(range(85,89))+[100], 'gani': list(range(89,90)),
                    'trigger': list(range(90,94))+[101], 'cycle_date': list(range(102,110))}

MED_CODE = {'folli: 75.0': 300, 'folli: 75.0 eod': 301, 'folli: 150.0': 302, 'folli: 150.0 eod': 303,
             'folli: 225.0': 304, 'folli: 225.0 eod': 305,  'folli: rhinal spray 6eod': 306, 'folli: no dose': 307,
             'menopur: 75.0': 308, 'menopur: 150.0': 309, 'menopur: 225.0': 310, 'menopur: no dose': 311,
             'clomid: 50.0': 312, 'clomid: 100.0': 313, 'clomid: no dose': 314, 'fem: 2.5': 315, 'fem: 5.0': 316,
             'fem: no dose': 317, 'orli: 0.125': 318, 'orli: 0.125 eod': 319, 'orli: 0.25': 320, 'orli: 0.25 eod': 321,
             'orli: no dose': 322, 'ocp: 0.5': 323, 'ocp: 0.5 eod': 324, 'ocp: 1.0': 325, 'ocp: 1.0 eod': 326, 'ocp: no dose': 327,
             'est: 1.0': 328, 'est: 2.0': 329, 'est: 4.0': 330, 'est: 6.0': 331, 'est po2pv2': 332, 'est: no dose': 333,
             'prog: 400.0': 334, 'prog: pv400 bid': 335, 'prog: pv400 po400': 336, 'prog: no dose': 337, 'GANI: 200': 338,
             'ovidrel': 339, 'lupron': 340, 'ovidrel+lupron': 341, 'no trigger': 342, 'too early to trigger': 350, 'folli partially missed': 343, 'clomid partially missed': 344,
             'fem partially missed': 345, 'orli partially missed': 346, 'ocp partially missed': 347, 'est partially missed' : 348, 
             'prog partially missed': 349}

MED_NODOSE_CODE = {'folli': 'folli: no dose', 'menopur': 'menopur: no dose', 'clomid': 'clomid: no dose',  
                    'fem': 'fem: no dose', 'orli': 'orli: no dose', 'ocp': 'ocp: no dose', 'est': 'est: no dose',
                    'prog': 'prog: no dose', 'No Trigger': 'no trigger'}

MED_NODOSE_CODE_ALT = {'ocp': 'ocp: no dose', 'est': 'est: no dose'}

DEMOGRAFIC_CODE = {'age: <=34': 500, 'age: 35-37': 501, 'age: 38-40': 502, 'age: >40': 503, 'BMI: unknown': 504, 'BMI: <18': 505, 
                    'BMI: 18-24': 506, 'BMI: 25-30': 507, 'BMI: >30': 508, 'menstruation cycle length: regular': 509, 
                    'menstruation cycle length: long': 510, 'menstruation cycle length: short': 511, 'r: 1-5': 512, 
                    'r: 6-10': 513, 'r: 11-15': 514, 'l: 1-5': 515, 'l: 6-10': 516, 'l: 11-15': 517}

CYCLE_DAY_CODE = {'cycle day: 0-3': 600, 'cycle day: 4-7': 601, 'cycle day: 8-11': 602, 'cycle day: 12-15': 603, 'cycle day: 16-19': 604, 
                    'cycle day: 20-23': 604, 'cycle day: 24-27': 605, 'cycle day: 28-': 606}

MED_MAP = {'OCP': 'ocp', 'GANI_CETR': 'orli', 'CC': 'clomid', 'Fem': 'fem', 'EST': 'est', 'PROG': 'prog', 'FSH_HMG': 'folli'}

MED_REV_MAP = {'OCP': ['OCP'], 'GANI_CETR': ['GANI', 'CENR', 'GANI/ORLI','ORLI'], 
                'CC': ['CC'], 'Fem': ['Fem', 'Letrozole'], 'EST': ['EST'], 
                'PROG': ['PROG'], 'FSH_HMG': ['Folli','FSH', 'HMG', 'Gonal'],
                'Trigger': ['NS/OVI', 'OVI', 'LUP', 'LUP/OVI', 'HCG', 'NS']}

RARE_DOSE_CONVERTER = {'folli: 300.0': 'folli: 225.0', 'folli: 300.0 eod': 'folli: 225.0 eod', 
                        'orli: 0.5': 'orli: 0.25', 'orli: 0.5 eod': 'orli: 0.25 eod', 
                        'orli: 0.333': 'orli: 0.25', 'orli: 0.333 eod': 'orli: 0.25 eod',
                        'clomid: 25.0': 'clomid: 50.0', 'clomid: 25.0 eod': 'clomid: 50.0 eod',
                        'est: 2.0 eod': 'est: 2.0', 'fem: 1.25': 'fem: 2.5', 'clomid: 12.5': 'clomid: 50.0',
                        'folli: 100.0': 'folli: 75.0', 'est: 1.5': 'est: 2.0', 'clomid: 150.0': 'clomid: 100.0', 
                        'est: 4.0 eod': 'est: 4.0'}

STIM_STARTER_ANCHOR = ['folli: 75.0', 'folli: 75.0 eod', 'folli: 150.0', 'folli: 150.0 eod', 'folli: 225.0', 'folli: 225.0 eod']

TRIGGER_RELATED = ['lupron', 'ovidrel', 'ovidrel+lupron', 'no trigger']

RESULT_RELATED = ['mii_label', 'twopn_label', 'blst_label']

TRIGGER_DAY_SEQ_LEN = 7

BLAST_RATE_KEY_TOKENS = ['cycle day:', 'e2:', 'fsh:', 'lh:', 'p4:', 'sono:', 'ovidrel', 'lupron', 'no trigger']

MISS_MED_PATTERN = 'Miss Med'

SOCIATY_CODE = {'no show': 700, 'pt decision': 701, 'miss med': 702, 'donor': 703, 'gc': 704}

CORE_EMBEDDING_CODE = {**TEST_RESULT_CODE, **MED_CODE, **CYCLE_DAY_CODE}

CORE_EMBEDDING_CODE_VOC = ['[UNK]', '[CLS]', '[SEP]', '[MASK]', '[PAD]'] +list(dict(sorted(CORE_EMBEDDING_CODE.items(), key=lambda item: item[1])).keys())

CODE_FOR_BASELINE = {**TEST_RESULT_CODE, **MED_CODE, **CYCLE_DAY_CODE, **DEMOGRAFIC_CODE}

CODE_FOR_BASELINE_VOC = ['[UNK]', '[CLS]', '[SEP]', '[MASK]', '[PAD]'] +list(dict(sorted(CODE_FOR_BASELINE.items(), key=lambda item: item[1])).keys())

TREATMENT_PRED_CODE = ['folli', 'orli', 'clomid', 'fem',  'ocp', 'est', 'trigger']

NEXT_VISIT_PRED_CODE = ['cycleday', 'e2', 'fsh', 'lh', 'p4', 'bhcg', 'sono', 'endo']

PRED_CODE = TREATMENT_PRED_CODE + NEXT_VISIT_PRED_CODE

DEPLOY_CODE = {**TREATMENT_CODE_INDEX, **NEXT_VISIT_INDEX}

DEMOGRAFIC_EMBEDDING_CODE_VOC = ['[UNK]', '[PAD]'] + list(DEMOGRAFIC_CODE.keys())

TEST_START_DATE = '2022-06-04'

NO_TRIGGER_FLAG = 'No Trigger'

TOO_EARLY_CODE = 'too early to trigger'

BLAST_RESULT_MAP = {'BLST: <40% BLST': 0,
                    'BLST: [40%, 60%] BLST': 1,
                    'BLST: >60% BLST': 2}

TWOPN_RESULT_MAP = {'2PN: <=60% 2PN': 0,
                    '2PN: (60%, 90%] 2PN': 1,
                    '2PN: >90% 2PN': 2}

MII_RESULT_MAP = {'MII: <=95% MII': 0,
                    'MII: >95% MII': 1}


def encode_age(age):
    assert 100 > age >12
    if age <= 34:
        return 'age: <=34'
    if 37>= age > 34:
        return 'age: 35-37'
    if 40>= age > 37:
        return 'age: 38-40'
    if age > 40:
        return 'age: >40'

def encode_bmi(bmi):
    if bmi < 9 or bmi > 65:
        return 'BMI: unknown'
    if bmi < 18:
        return 'BMI: <18'
    if  24 >= bmi >= 18:
        return 'BMI: 18-24'
    if  30 >= bmi >= 25:
        return 'BMI: 25-30'
    if  bmi > 20:
        return 'BMI: >30'

def encode_cycle_len(cycle_len):
    if cycle_len < 0:
        return 'menstruation cycle length: regular'
    if cycle_len < DEFAULT_CYCLE_LENGTH:
        return 'menstruation cycle length: short'
    if cycle_len == DEFAULT_CYCLE_LENGTH:
        return 'menstruation cycle length: regular'
    if cycle_len > DEFAULT_CYCLE_LENGTH:
        return 'menstruation cycle length: long'


def encode_folli(follicles):
    if not follicles:
        return '', [] 
    _count_20, _count_15, _count_10, _count_afs, _count_all = 0, 0, 0, 0, 0
    _raw_folli_dict = {}
    _rpt_list = []
    for side in follicles:
        _raw_folli_dict[side] = {}
        for folli_str in follicles[side].split(", "):
            if folli_str == '':
                continue
            if folli_str not in _raw_folli_dict[side]:
                _raw_folli_dict[side][folli_str] = 1
            else:
                _raw_folli_dict[side][folli_str] += 1

    if sum(len(sub_d) for sub_d in _raw_folli_dict.values()) == 0: 
        return '', []

    for side in _raw_folli_dict:
        for folli_str in _raw_folli_dict[side]:
            try:
                if float(folli_str) >= FAKE_FOLLI_BOUND and _raw_folli_dict[side][folli_str] >= FAKE_FOLLI_DUPLICATE_BOUND:
                    _rpt_list.append('%s, %s' % (folli_str, _raw_folli_dict[side][folli_str]))
                else:
                    if float(folli_str) >= 20:
                        _count_20 += _raw_folli_dict[side][folli_str]
                    if float(folli_str) >= 15:
                        _count_15 += _raw_folli_dict[side][folli_str]
                    if float(folli_str) > 10:
                        _count_10 += _raw_folli_dict[side][folli_str]
                    if 0 < float(folli_str) <= 10:
                        _count_afs += _raw_folli_dict[side][folli_str]
                    _count_all += _raw_folli_dict[side][folli_str]
            except:
                pass

    if _count_all == 0:
        return 'sono: quiet', _rpt_list

    if _count_afs == _count_all:
        if _count_afs <= 5:
            return 'afs: 1-5', _rpt_list
        if _count_afs <= 10:
            return 'afs: 6-10', _rpt_list
        if _count_afs <= 20:
            return 'afs: 10-20', _rpt_list
        return 'afs: 20+', _rpt_list
        
        
    if _count_20 / _count_all >= 0.15 and _count_15 / _count_all >= 0.35:
        return 'sono: >20: >15% and >15: >35%', _rpt_list
    if _count_20 / _count_all >= 0.15 and _count_15 / _count_all < 0.35:
        return 'sono: >20: >15% and >15: <35%', _rpt_list
    if _count_20 / _count_all < 0.15 and _count_15 / _count_all >= 0.35:
        return 'sono: >20: 0-15% and >15: >35%', _rpt_list
    if _count_20 / _count_all < 0.15 and _count_15 / _count_all < 0.35:
        return 'sono: >20: 0-15% and >15: <35%', _rpt_list
    if _count_20 == 0 and _count_15 / _count_all >= 0.35 and _count_10 / _count_all >= 0.35:
        return 'sono: >15: >35% and >8: <35%', _rpt_list
    if _count_20 == 0 and _count_15 / _count_all < 0.35 and _count_10 / _count_all >= 0.35:
        return 'sono: >15: >35% and >8: >35%', _rpt_list

    print('%s %s %s %s' % (_count_20, _count_15, _count_10, _count_afs, _count_all))
    return '', _rpt_list

def encode_cyst(cyst):
    if not cyst:
        return ''
    _max = 0
    for side in cyst:
        try:
            _max = max(_max, max([float(c) for c in cyst[side].split(', ')]))
        except:
            pass
    if _max > 15:
        return 'cyst: > 15'
    return ''

def encode_endo(endo):
    if endo < 0:
        return ''
    if endo < 5:
        return 'endo: <5'
    if 7 > endo >=5:
        return 'endo: 5-7'
    if 13 > endo >=7:
        return 'endo: 7-13'
    if endo >=13:
        return 'endo: 13-'

def encode_fsh(fsh):
    try:
        if fsh < 0:
            return ''
        if fsh <= 5:
            return 'fsh: 0-5'
        if 15 >= fsh > 5:
            return 'fsh: 5-15'
        if 30 >= fsh > 15:
            return 'fsh: 15-30'
        if 40 >= fsh > 30:
            return 'fsh: 30-40' 
        if fsh > 40:
            return 'fsh: 40-'   
        return ''
    except:
        return ''

def encode_e2(e2):
    try:
        if e2 < 0:
            return ''
        if e2 <= 50:
            return 'e2: 0-50'
        if 100 >= e2 > 50:
            return 'e2: 50-100'
        if 200 >= e2 > 100:
            return 'e2: 100-200'
        if 500 >= e2 > 200:
            return 'e2: 200-500' 
        if 1000 >= e2 > 500:
            return 'e2: 500-1000'   
        if 1500 >= e2 > 1000:
            return 'e2: 1000-1500' 
        if 2000 >= e2 > 1500:
            return 'e2: 1500-2000' 
        if 3000 >= e2 > 2000:
            return 'e2: 2000-3000' 
        if e2 >3000:
            return 'e2: 3000-'  
        
    except:
        return ''

def encode_lh(lh):
    try:
        if lh < 0:
            return ''
        if lh <= 1:
            return 'lh: 0-1'
        if 1.5 >= lh > 1:
            return 'lh: 1-1.5'
        if 2 >= lh > 1.5:
            return 'lh: 1.5-2'
        if 3 >= lh > 2:
            return 'lh: 2-3' 
        if 5 >= lh > 3:
            return 'lh: 3-5'   
        if 7 >= lh > 5:
            return 'lh: 5-7' 
        if 10 >= lh > 7:
            return 'lh: 7-10' 
        if lh > 10:
            return 'lh: 10-'  
        return ''
    except:
        return ''

def encode_p4(p4):

    try:
        if p4 < 0:
            return ''
        if p4 <= 0.5:
            return 'p4: 0-0.5'
        if 1 >= p4 > 0.5:
            return 'p4: 0.5-1'
        if 1.5 >= p4 > 1:
            return 'p4: 1-1.5'
        if 2 >= p4 > 1.5:
            return 'p4: 1.5-2' 
        if p4 > 2:
            return 'p4: 2-'   
        return ''
    except:
        return ''

def encode_hcg(hcg):
    try:
        if hcg < 0:
            return ''
        if hcg < 5:
            return 'bhcg: 0-5'
        if hcg > 5:
            return 'bhcg: 5-'
        return ''
    except:
        return ''

def encode_cycle_day(cycle_day):
    if cycle_day <= 3:
        return 'cycle day: 0-3'
    if 7 >= cycle_day >= 4:
        return 'cycle day: 4-7'
    if 11 >= cycle_day >= 8:
        return 'cycle day: 8-11'
    if 15 >= cycle_day >= 12:
        return 'cycle day: 12-15'
    if 19 >= cycle_day >= 16:
        return 'cycle day: 16-19'
    if 23 >= cycle_day >= 20:
        return 'cycle day: 20-23'
    if 27 >= cycle_day >= 24:
        return 'cycle day: 24-27'
    if cycle_day >= 28:
        return 'cycle day: 28-'
