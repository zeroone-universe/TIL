def get_paths_wav(paths):
    wav_paths = []
    if type(paths) == str:
        paths = [paths]
    
    # 1. get wav paths
    for path in paths:
        for root, dirs, files in os.walk(path):
            
            wav_paths += [os.path.join(root, file) for file in files if os.path.splitext(file)[-1] == '.wav']
    
    # 2. sort by filenames
    wav_paths.sort(key = lambda x: os.path.split(x)[-1])