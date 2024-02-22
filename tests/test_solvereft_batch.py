from test_solvereft import main
from dabry.problem import NavigationProblem

if __name__ == '__main__':
    pb_names = [k for k, v in NavigationProblem.ALL.items() if v['gh_wf'] == 'True']
    for pb_name in pb_names:
        print(pb_name)
        main(pb_name)
