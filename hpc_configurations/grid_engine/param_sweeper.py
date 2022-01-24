import os

# SBATCH fields assumed if the particular field isn't passed to get_script
# can contain unused fields

DEFAULT_QSUB_FIELDS = {
    'sge_memory': 64,
    'sge_memory_unit': 'GB',
    'sge_num_nodes': 1,
    'sge_num_cpus': 10,
    'sge_num_gpus': 1,
    'sge_time_d': 0,
    'sge_time_h': 10,
    'sge_time_m': 0,
    'sge_time_s': 0,
    'sge_job_name': 'myjob',
    'sge_cluster_name': 'mycluster',
    'sge_entry_point': 'train.py',
}

# a template for the entire submit script
# (bash braces must be escaped by doubling: $var = ${{var}})
# num_jobs, param_arr_init, param_val_assign and param_list are special fields

TEMPLATE = '''
#!/bin/env bash
#SBATCH --array=0-{num_jobs}
#SBATCH --job-name={sge_job_name}
#SBATCH --output=logs/{sge_job_name}-%a.out
#SBATCH --error=logs/{sge_job_name}-%a.err
#SBATCH --mem={sge_memory}{sge_memory_unit}
#SBATCH --time={sge_time_d}-{sge_time_h}:{sge_time_m}:{sge_time_s}
#SBATCH --nodes={sge_num_nodes}
#SBATCH --partition={sge_cluster_name}
#SBATCH --cpus-per-task={sge_num_cpus}
#SBATCH --gpus-per-node={sge_num_gpus}

{param_arr_init}
trial=${{SBATCH_ARRAY_TASK_ID}}
{param_val_assign}

module purge
module load anaconda3
conda activate poetry

export PYTHONUNBUFFERED=1

poetry shell

python {sge_entry_point} {script_args}

'''.strip()


# functions for making bash expressions
# bash braces are escaped by doubling

def _mth(exp):
    return '$(( %s ))' % exp


def _len(arr):
    return '${{#%s[@]}}' % arr


def _get(arr, elem):
    return '${{%s[%s]}}' % (arr, elem)


def _eq(var, val):
    return '%s=%s' % (var, val)


def _op(a, op, b):
    return _mth('%s %s %s' % (a, op, b))


def _arr(arr):
    return '( %s )' % ' '.join(map(str, arr))


def _seq(a, b, step):
    return '($( seq %d %d %d ))' % (a, step, b)


def _var(var):
    return '${%s}' % var


# templates for param array construction and element access

PARAM_ARR = '{param}_values'
PARAM_EXPRS = {
    'param_arr_init':
        _eq(PARAM_ARR, '{values}'),
    'param_val_assign': {
        'assign':
            _eq('{param}', _get(PARAM_ARR, _op('trial', '%', _len(PARAM_ARR)))),
        'increment':
            _eq('trial', _op('trial', '/', _len(PARAM_ARR)))
    }
}


def _to_bash(obj):
    if isinstance(obj, range):
        return _seq(obj.start, obj.stop - 1, obj.step)
    if isinstance(obj, list) or isinstance(obj, tuple):
        return _arr(obj)
    raise ValueError('Unknown object type %s' % type(obj).__name__)


def _get_params_bash(params, values):
    # builds bash code to perform the equivalent of
    '''
    def get_inds(params, ind):
        inds = []
        for length in map(len, params):
            inds.append(ind % length)
            ind //= length
        return inds[::-1]
    '''

    # get lines of bash code for creating/accessing param arrays
    init_lines = []
    assign_lines = []
    init_temp = PARAM_EXPRS['param_arr_init']
    assign_temps = PARAM_EXPRS['param_val_assign']

    for param, vals in zip(params, values):
        param = param.replace('-', '_')
        init_lines.append(
            init_temp.format(param=param, values=_to_bash(vals)))
        assign_lines.append(
            assign_temps['assign'].format(param=param))
        assign_lines.append(
            assign_temps['increment'].format(param=param))

    # remove superfluous final trial reassign
    assign_lines.pop()

    return init_lines, assign_lines


def get_script(fields, params, param_order=None):
    '''
    returns a string of a SLURM submission script using the passed fields
    and which creates an array of jobs which sweep the given params
    fields:      dict of SLURM field names to their values. type is ignored
    params:      a dict of (param names, param value list) pairs.
                 The param name is the name of the bash variable created in
                 the submission script which will contain the param's current
                 value (for that SLURM job instance). param value list is
                 a list (or range instance) of the values the param should take,
                 to be run once against every other possible configuration of all params.
    param_order: a list containing all param names which indicates the ordering
                 of the params in the sweep. The last param changes every
                 job number. If not supplied, uses an arbitrary order
    '''

    # check arguments have correct type
    assert isinstance(fields, dict)
    assert isinstance(params, dict)
    assert (isinstance(param_order, list) or
            isinstance(param_order, tuple) or
            param_order == None)
    if param_order == None:
        param_order = list(params.keys())

    # check each field appears in the template
    for field in fields:
        if ('{%s}' % field) not in TEMPLATE:
            raise ValueError('passed field %s unused in template' % field)

    # calculate total number of jobs (minus 1; SLURM is inclusive)
    num_jobs = 1
    for vals in params.values():
        num_jobs *= len(vals)
    num_jobs -= 1

    # get bash code for param sweeping
    init_lines, assign_lines = _get_params_bash(
        param_order, [params[key] for key in param_order])

    # build args
    script_args = ''
    for key in param_order:
        script_args += f' --{key}=${{{key.replace("-", "_")}}}'

    # build template substitutions (overriding defaults)
    subs = {
        'param_arr_init': '\n'.join(init_lines),
        'param_val_assign': '\n'.join(assign_lines),
        'param_list': ', '.join(map(_var, param_order)),
        'num_jobs': num_jobs,
        'script_args': script_args
    }
    for key, val in DEFAULT_QSUB_FIELDS.items():
        subs[key] = val
    for key, val in fields.items():
        subs[key] = val

    return TEMPLATE.format(**subs)


def save_script(filename, fields, params, param_order=None):
    '''
    creates and writes to file a SLURM submission script using the passed
    fields and which creates an array of jobs which sweep the given params
    fields:      dict of SLURM field names to their values. type is ignored
    params:      a dict of (param names, param value list) pairs.
                 The param name is the name of the bash variable created in
                 the submission script which will contain the param's current
                 value (for that SLURM job instance). param value list is
                 a list (or range instance) of the values the param should take,
                 to be run once against every other possible configuration of all params.
    param_order: a list containing all param names which indicates the ordering
                 of the params in the sweep. The last param changes every
                 job number. If not supplied, uses an arbitrary order
    '''

    script_str = get_script(fields, params, param_order)
    if ('/' in filename) or ('\\' in filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write(script_str)


if __name__ == '__main__':
    # A simple example for griddly clusters environment
    script = get_script(
        {
            'sge_time_h': 10,
            'sge_num_gpus': 1,
            'sge_num_cpus': 10,
            'sge_job_name': 'sge_test'
        },
        {

            'gym-id': ['GDY-Clusters-0'],
            'exp-name': ['griddly-clusters-sge-sweep'],
            'track': ['True'],
            'total-timesteps': [50000000],
            'processes': [8],
            'num-envs': [128],
            'learning-rate': [0.005, 0.001, 0.0005],
            'ent-coef': [0.2, 0.1, 0.05],
            'eval-interval': [50000],
            'eval-steps': [300],
            'eval-num-env': [4]
        })
    print(script)
