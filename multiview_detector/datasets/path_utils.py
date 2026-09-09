import os
from collections import deque


_KAGGLE_SEARCH_ROOTS = ('/kaggle/input', '/kaggle/working')
_SKIP_DIRS = {'Image_subsets', 'annotations_positions', '.git', '__pycache__'}

# Structural signatures used to distinguish full or partially annotated copies
# of the two supported datasets. The contents of annotations_positions may be
# partial; only the directory and the original camera calibration layout matter.
_DATASET_SIGNATURES = {
    'wildtrack': (
        'Image_subsets',
        'annotations_positions',
        os.path.join('calibrations', 'intrinsic_zero', 'intr_CVLab1.xml'),
        os.path.join('calibrations', 'extrinsic', 'extr_CVLab1.xml'),
    ),
    'multiviewx': (
        'Image_subsets',
        'annotations_positions',
        os.path.join('calibrations', 'intrinsic', 'intr_Camera1.xml'),
        os.path.join('calibrations', 'extrinsic', 'extr_Camera1.xml'),
    ),
}


def _is_dataset_root(path, required_paths):
    return os.path.isdir(path) and all(
        os.path.exists(os.path.join(path, relative_path))
        for relative_path in required_paths
    )


def _find_dataset_root(search_root, required_paths, max_depth=4):
    """Search shallowly without descending into large image/annotation folders."""
    if not os.path.isdir(search_root):
        return None

    queue = deque([(os.path.abspath(search_root), 0)])
    while queue:
        current, depth = queue.popleft()
        if _is_dataset_root(current, required_paths):
            return current
        if depth >= max_depth:
            continue

        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name.lower())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False) and entry.name not in _SKIP_DIRS:
                queue.append((entry.path, depth + 1))
    return None


def _normalized_dir_name(name):
    return ''.join(character for character in name.lower() if character.isalnum())


def _find_named_dirs(search_root, names, max_depth=4):
    """Find shallow directories by name without entering large data folders."""
    if not search_root or not os.path.isdir(search_root):
        return []

    normalized_names = {_normalized_dir_name(name) for name in names}
    matches = []
    queue = deque([(os.path.abspath(search_root), 0)])
    while queue:
        current, depth = queue.popleft()
        if _normalized_dir_name(os.path.basename(current)) in normalized_names:
            matches.append(current)
        if depth >= max_depth:
            continue

        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name.lower())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False) and entry.name not in _SKIP_DIRS:
                queue.append((entry.path, depth + 1))
    return matches


def detect_dataset_root(root, max_depth=4, dataset_name=None):
    """Return ``(dataset_name, dataset_root)`` detected below ``root``.

    The supplied path may be the dataset root itself or a shallow parent such
    as a Kaggle input directory containing an extra dataset/version folder.
    When ``dataset_name`` is supplied, only that dataset is considered. This
    makes a shared parent containing both Wildtrack and MultiviewX unambiguous.
    """
    requested_root = os.path.abspath(os.path.expanduser(os.fspath(root)))
    if not os.path.isdir(requested_root):
        raise FileNotFoundError(f'Dataset path does not exist or is not a directory: {requested_root}')

    if dataset_name is not None and dataset_name not in _DATASET_SIGNATURES:
        choices = ', '.join(_DATASET_SIGNATURES)
        raise ValueError(f'Unsupported dataset {dataset_name!r}. Choose one of: {choices}')

    signatures = (
        {dataset_name: _DATASET_SIGNATURES[dataset_name]}
        if dataset_name is not None
        else _DATASET_SIGNATURES
    )

    direct_matches = [
        (dataset_name, requested_root)
        for dataset_name, required_paths in signatures.items()
        if _is_dataset_root(requested_root, required_paths)
    ]
    if len(direct_matches) == 1:
        return direct_matches[0]

    matches = []
    for dataset_name, required_paths in signatures.items():
        resolved_root = _find_dataset_root(requested_root, required_paths, max_depth=max_depth)
        if resolved_root is not None:
            matches.append((dataset_name, resolved_root))

    if not matches:
        expected = '; '.join(
            f'{name}: {", ".join(paths)}'
            for name, paths in signatures.items()
        )
        raise FileNotFoundError(
            f'Could not detect a Wildtrack or MultiviewX dataset under {requested_root}. '
            f'Expected one of these layouts: {expected}'
        )
    if len(matches) > 1:
        found = ', '.join(f'{name} at {path}' for name, path in matches)
        raise ValueError(
            f'Dataset path is ambiguous because it contains multiple datasets: {found}. '
            'Pass --data_path pointing to one dataset root.'
        )
    return matches[0]


def resolve_annotation_dirs(
        dataset_root,
        dataset_name,
        partial_annotation_percent,
        search_root=None,
        dropped_path=None,
):
    """Resolve the observed and hidden annotation directories for one run.

    Images, calibration files, and evaluation ground truth always stay under
    ``dataset_root``.  For a partial-annotation run, only the annotations used
    to construct training targets are read from the sibling dropped dataset:

    ``<dataset>_dropped/drop<percent>/{annotations_positions,hidden_annotations_positions}``

    ``dropped_path`` may point to a separate mounted input containing the
    dropped dataset, the ``<dataset>_dropped`` directory itself, or the
    selected ``drop<percent>`` directory. When it is omitted, resolution falls
    back to the sibling of the complete dataset and ``search_root`` for
    backward compatibility. No machine-specific dataset path is embedded here.
    """
    if not 0 <= partial_annotation_percent < 100:
        raise ValueError('partial annotation percentage must be in [0, 100)')

    dataset_root = os.path.abspath(os.path.expanduser(os.fspath(dataset_root)))
    full_annotation_dir = os.path.join(dataset_root, 'annotations_positions')
    if not os.path.isdir(full_annotation_dir):
        raise FileNotFoundError(f'Missing full annotation directory: {full_annotation_dir}')
    if partial_annotation_percent == 0:
        return full_annotation_dir, None

    setting = f'drop{partial_annotation_percent}'
    parent = os.path.dirname(dataset_root)
    dropped_root_names = (
        f'{os.path.basename(dataset_root)}_dropped',
        f'{dataset_name}_dropped',
    )
    candidates = []

    def add_candidate(path):
        path = os.path.abspath(path)
        if path not in candidates:
            candidates.append(path)

    if dropped_path is not None:
        requested_dropped_path = os.path.abspath(
            os.path.expanduser(os.fspath(dropped_path))
        )
        if not os.path.isdir(requested_dropped_path):
            raise FileNotFoundError(
                f'Dropped annotation path does not exist or is not a directory: '
                f'{requested_dropped_path}'
            )
        if _normalized_dir_name(os.path.basename(requested_dropped_path)) == setting:
            add_candidate(requested_dropped_path)
        add_candidate(os.path.join(requested_dropped_path, setting))
        for dropped_root in _find_named_dirs(requested_dropped_path, dropped_root_names):
            add_candidate(os.path.join(dropped_root, setting))
    else:
        for dropped_root_name in dropped_root_names:
            add_candidate(os.path.join(parent, dropped_root_name, setting))

        if search_root is not None:
            requested_root = os.path.abspath(os.path.expanduser(os.fspath(search_root)))
            for dropped_root in _find_named_dirs(requested_root, dropped_root_names):
                add_candidate(os.path.join(dropped_root, setting))

    for setting_root in candidates:
        annotation_dir = os.path.join(setting_root, 'annotations_positions')
        hidden_annotation_dir = os.path.join(setting_root, 'hidden_annotations_positions')
        if os.path.isdir(annotation_dir) and os.path.isdir(hidden_annotation_dir):
            return annotation_dir, hidden_annotation_dir

    checked = ', '.join(candidates)
    raise FileNotFoundError(
        f'Could not find dropped annotations for --pa {partial_annotation_percent}. '
        f'Checked: {checked}.'
    )


def resolve_dataset_root(root, dataset_name, required_paths):
    """Resolve a local dataset path, falling back to Kaggle-mounted datasets."""
    requested_root = os.path.abspath(os.path.expanduser(os.fspath(root)))
    if _is_dataset_root(requested_root, required_paths):
        return requested_root

    for search_root in _KAGGLE_SEARCH_ROOTS:
        resolved_root = _find_dataset_root(search_root, required_paths)
        if resolved_root is not None:
            return resolved_root

    required = ', '.join(required_paths)
    raise FileNotFoundError(
        f'Could not find the {dataset_name} dataset. Checked {requested_root} and '
        f'{", ".join(_KAGGLE_SEARCH_ROOTS)}. Expected a directory containing: {required}'
    )
