class CreditsParser:
    def __init__(self, credits_data):
        self.credits_data = credits_data
        self.cast = []
        self.crew = []
        self.companies = []
        
    def parse(self):
        self._parse_cast()
        self._parse_crew()
        self._parse_companies()
        return {
            'cast': self.cast,
            'crew': self.crew,
            'companies': self.companies
        }
        
    def _parse_cast(self):
        # heuristic logic for cast separation
        for entry in self.credits_data['cast']:
            name = self.detect_name(entry)
            self.cast.append({'name': name, 'role': self.normalize_role(entry['role'])})
        
    def _parse_crew(self):
        # heuristic logic for crew separation
        for entry in self.credits_data['crew']:
            name = self.detect_name(entry)
            self.crew.append({'name': name, 'role': self.normalize_role(entry['role'])})
        
    def _parse_companies(self):
        # heuristic logic for company separation
        for company in self.credits_data['companies']:
            name = self.detect_company(company)
            self.companies.append({'company': name})
        
    def normalize_role(self, role):
        # Normalization logic from a predefined JSON file (credits_role_alias_tr.json)
        # This would typically load a file and return a canonical role based on input
        role_dict = self.load_role_aliases()
        return role_dict.get(role, role)

    def detect_name(self, entry):
        # Logic to detect and return names from entry
        return entry.get('name', '')

    def detect_company(self, company):
        # Logic to detect and return company names
        return company.get('name', '')

    def load_role_aliases(self):
        # Load role aliases from JSON
        return {"Director": "Director", "Producer": "Producer"}  # Placeholder
