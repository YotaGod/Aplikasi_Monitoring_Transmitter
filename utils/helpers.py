from utils.normal_ranges import NORMAL_RANGES

def get_status(value, param_name):
    """Menentukan status berdasarkan nilai parameter (nilai asli)"""
    ranges = NORMAL_RANGES.get(param_name, {})
    if not ranges:
        return 'normal', '🟢'
    
    min_val = ranges.get('min', -float('inf'))
    max_val = ranges.get('max', float('inf'))
    
    range_width = max_val - min_val
    warning_min = min_val + range_width * 0.15
    warning_max = max_val - range_width * 0.15
    
    if value < min_val or value > max_val:
        return 'danger', '🔴'
    elif value < warning_min or value > warning_max:
        return 'warning', '🟡'
    else:
        return 'normal', '🟢'

def get_global_status(df):
    """Menentukan status global dari data terbaru"""
    if len(df) == 0:
        return 'normal', '🟢', 'NORMAL'
    
    latest = df.iloc[-1]
    statuses = []
    for col in NORMAL_RANGES.keys():
        if col in latest:
            status, _ = get_status(latest[col], col)
            statuses.append(status)
    
    if 'danger' in statuses:
        return 'danger', '🚨', 'DANGER'
    elif 'warning' in statuses:
        return 'warning', '🟡', 'WARNING'
    else:
        return 'normal', '🟢', 'NORMAL'