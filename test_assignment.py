import nbformat
from pathlib import Path


def _load_intersection_func():
    nb_path = Path(__file__).parent / "test-assignment.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            src = cell.get("source", [])
            code = src if isinstance(src, str) else "\n".join(src)
            exec(code, ns)
    func = ns.get("intersection")
    if func is None or not callable(func):
        raise AssertionError("Notebook must define a callable function named intersection(array1, array2)")
    return func


def test_example_case():
    intersection = _load_intersection_func()
    assert intersection([3, 9, 5, 7], [2, 5, 9]) in ([5, 9], [9, 5])


def test_handles_duplicates_and_order_irrelevant():
    intersection = _load_intersection_func()
    result = intersection([1, 1, 2, 3], [3, 3, 1])
    assert set(result) == {1, 3}


def test_no_overlap_returns_empty():
    intersection = _load_intersection_func()
    assert intersection([4, 5], [6, 7]) == []
