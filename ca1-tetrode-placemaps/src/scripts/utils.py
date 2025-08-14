import os
import fnmatch
import pathlib


EXCLUDE_DIRS = (".phy", )

def get_recording_base_files(base_dir, pattern, full_search=True):
    """More efficient recursive directory searching than `rglob`
    by explicitly excluding the horrific maze of Kilosort/phy output
    
    Since rglob has no way to exclude whole subdirectories, searching
    the.phy output can add a minute (locally) or *hours* (via VPN) to 
    total search time
    
    os.walk is recursive based on the list `dirs` returned by itself. 
    By proactively deleting entries in this list, sub-directories can
    be explicitly excluded from the search, thus cutting down total
    search time. Initially, only the `.phy` dir is excluded, but multiple
    exclusions are supported
    
    Parameters
    ----------
    base_dir : path-like
        The location to start the search from. 
        e.g. "/mnt/N/neuropixels/data/26230/20191204"
    pattern: str
        pattern that matches the files that are desired
        e.g. "*.ap.meta"
    full_search: bool, optional
        Continue searching for additional candidates after a single matching
        candidate is found?
        Default True
    
    Returns
    -------
    list of path-like
        List of absolute paths matching the pattern
        May be empty if no candidates are found. 
    """
    output = []
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            if fnmatch.fnmatch(file_name, pattern):
                full_path = pathlib.Path(root, file_name)
                output.append(full_path)
            if not full_search:
                return output
        for exclude in EXCLUDE_DIRS:
            for dir_name in dirs:
                if fnmatch.fnmatch(dir_name, exclude):
                    dirs.remove(dir_name)
                    break
                    # To speed up matching, it will only ever exclude the *first*
                    # matching hit in dirs. If many similar dirs should be
                    # excluded, this will need rewriting, or EXCLUDE_DIRS
                    # must be made more specific.
    return output


def handle_string(value):
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
    return value

def parse_rate(rate_string):
    sample_rate, sample_unit = rate_string.split(" ")
    assert sample_unit.lower() == "hz", 'Sampling rate must be given in "hz"'
    return float(sample_rate)


def advance_until(f, search_string):
    """
    Advances file cursor until the end of the `search_string`
    Args:
        f: Opened binary file handle
        search_string: string to search for

    Returns:
        Data upto and including the `search_string` if `search_string` found. None otherwise.

    """
    # TODO: handle other encoding
    len_search = len(search_string)
    data = ''
    while True:
        b = f.read(1)
        if not b:
            return None
        data += b.decode('latin1')
        if data[-len_search:] == search_string:
            return data
