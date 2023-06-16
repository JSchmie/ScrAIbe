import json

ALPHABET = [*"abcdefghijklmnopqrstuvwxyz"]


class Transcript:
    """
    Class for storing transcript data
    and exporting it to files in different formats
    """
    def __init__(self, transcript: dict) -> None:
        """
        :param transcript: formated transcript string
        """
        self.transcript = transcript
        self.speakers = self._extract_speakers()
        self.segments = self._extract_segments()
        self.annotation = {}
    
    def annotate(self, *args, **kwargs) -> dict:
        """
        Annote transcript to define speaker names
        
        :param args: list of speaker names will maped sequentially to the speakers
        :param kwargs: dict with speaker names as keys and list of segments as values
        
        :return: dict with speaker names as keys and list of segments as values
        :rtype: dict
        """
        
        annotatios = {}

        if len(args) != len(self.speakers):
            raise ValueError("Number of speaker names does not match number of speakers")
        
        if args:
            for arg,ospeaker in zip(args,self.speakers):
                annotatios[ospeaker] = arg
        
        if kwargs:
            for key in kwargs:
                if key not in self.speakers:
                    raise ValueError(f"{key} is not a speaker")
                annotatios[key] = kwargs[key]

        self.annotation = annotatios
        return annotatios
    
    def _extract_speakers(self) -> list:
        """
        Extract speaker names from transcript
        :return: list of speaker names
        :rtype: list
        """
        return list(set([self.transcript[id]["speaker"] for id in self.transcript]))
    
    def _extract_segments(self) -> list:
        """
        Extract segments from transcript

        :return: list of segments
        :rtype: list
        """
        return [self.transcript[id]["segment"] for id in self.transcript]

    def __str__(self) -> str:
        """
        Get transcript as string

        :return: transcript as string
        :rtype: str
        """
        fstring = ""
        
        for id in self.transcript:
            seq = self.transcript[id]
            
            if self.annotation:
                speaker = self.annotation[seq["speaker"]]
            else:
                speaker = seq["speaker"]
                
            fstring += f"{speaker}: {seq['text']}\n"

        return fstring
    
    def __repr__(self) -> str:
        return f"Transcript(speakers = {self.speakers},"\
                f"segments = {self.segments}, annotation = {self.annotation})"
    
    def get_dict(self) -> dict:
        """
        Get transcript as dict

        :return: transcript as dict
        :rtype: dict
        """
        
        return self.transcript
    
    def get_json(self, *args, **kwargs) -> str:
        """
        Get transcript as json string
        :return: transcript as json string
        :rtype: str
        """
        if "indent" not in kwargs:
            kwargs["indent"] = 4
        return json.dumps(self.transcript, *args, **kwargs)
    
    def get_html(self) -> str:
        """
        Get transcript as html string

        :return: transcript as html string
        :rtype: str
        """
        html = "<p>" + self.__str__().replace("\n", "<br>") + "</p>"
        html = "<html><body>" + html + "</body></html>"
        html = html.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        
        return html
        
    
    def get_md(self) -> str:
        return self.get_html()
    
    def get_tex(self) -> str:
        
        if not self.annotation:

            self.annotate(*ALPHABET[:len(self.speakers)])
        
        fstring ="\\begin{drama}"
        
        for speaker in self.speakers:
            
            fstring += "\n\t\\Character{"+ str(self.annotation[speaker]) + "}" \
                "{"+ str(self.annotation[speaker]) + "}"
        
        for id in self.transcript:
            seq = self.transcript[id]
            speaker = self.annotation[seq["speaker"]]
            fstring += f"\n\\{speaker}speaks:\n{seq['text']}"
        
        fstring += "\n\\end{drama}"
        
        return fstring
        
            
    def to_json(self,path, *args, **kwargs) -> None:
        """
        Save transcript as json file
        :param path: path to save file
        :type path: str
        """
        with open(path, "w") as f:
            json.dump(self.transcript, f, *args, **kwargs)
    
    def to_txt(self, path: str) -> None:
        
       with open(path, "w") as f:
            f.write(self.__str__, f)
    
    def to_md(self, path: str) -> None:
        return self.to_html(path)
    
    def to_html(self, path: str) -> None:
        """
        Save transcript as html file

        :param path: path to save file
        :type path: str
        """
        
        with open(path, "w") as file:
            file.write(self.get_html())
    
    def to_tex(self, path: str) -> None:
        pass
    
    def to_pdf(self, path: str) -> None:
        pass
    
if __name__ == "__main__":
    test = Transcript(json.load(open("tests/test.json", "r")))
    print(repr(test))
    print(test)
    
    
    
    