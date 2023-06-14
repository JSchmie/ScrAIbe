
class Transcript:
    """
    Class for storing transcript data
    and exporting it to files in different formats
    """
    def __init__(self, transcript: str) -> None:
        """
        :param transcript: formated transcript string
        """
        self.transcript = transcript
    
    def to_latex(self, path: str) -> None:
        pass
    
    def to_pdf(self, path: str) -> None:
        pass
    
    def to_txt(self, path: str) -> None:
        pass
    
    def to_json(self, path: str) -> None:
        pass