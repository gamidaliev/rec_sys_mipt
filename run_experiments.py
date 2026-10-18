import argparse
import time
from pathlib import Path
import nbformat


def execute_in_process(notebook, path):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline')
    from matplotlib_inline.backend_inline import configure_inline_support
    configure_inline_support(shell, 'inline')
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        start = time.perf_counter()
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = count
        cell.outputs = []
        for name in ['stdout', 'stderr']:
            value = getattr(captured, name)
            if value:
                cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=value))
        for display in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=display.data, metadata=display.metadata))
        error = result.error_before_exec or result.error_in_exec
        if error:
            cell.outputs.append(nbformat.v4.new_output('error', ename=type(error).__name__, evalue=str(error), traceback=[str(error)]))
        nbformat.write(notebook, path)
        print(f'Cell {count}: {time.perf_counter() - start:.1f}s', flush=True)
        if error:
            raise error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-process', action='store_true', help='Execute without a separate Jupyter kernel')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    path = root / 'experiments.ipynb'
    notebook = nbformat.read(path, as_version=4)
    # Run from any working directory, using the repository as the notebook directory.
    import os
    os.chdir(root)
    if args.in_process:
        execute_in_process(notebook, path)
    else:
        from nbclient import NotebookClient
        client = NotebookClient(notebook, timeout=3600, kernel_name='python3',
                                resources={'metadata': {'path': str(root)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
    print('Saved:', path)
