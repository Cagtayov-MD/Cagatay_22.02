class OCRBlacklist:
    def __init__(self, languages):
        self.blacklist = set()
        self.languages = languages
    
    def add_to_blacklist(self, name):
        self.blacklist.add(name)

    def is_blacklisted(self, name):
        return name in self.blacklist

class NameSplitter:
    def __init__(self):
        pass  # Implementation for name splitting

    def split_name(self, full_name):
        # Simple example splitting logic, can be enhanced
        parts = full_name.split(' ')
        return parts[0], parts[-1]  # return first and last name

def process_frames(frames):
    # Assuming frames is a list of frames to be processed
    for frame in frames:
        # Integrate the new filtering pipeline here
        # _blacklist_filter_v2(frame)
        # _name_split_pass_v2(frame)
        pass  # Further processing logic here
