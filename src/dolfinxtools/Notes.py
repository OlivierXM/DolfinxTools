class Notes(object):
    """
        Provide general class for compiling notes during script development
    """
    def __init__(self):
        """
            Initialize Notes, argless
        """
        self.messages = []

    def addNotes(self, newMessage:str):
        """
            Add a new note to the list
            Args:
                newMessage: New message to add [str]
        """
        self.messages.append(newMessage)

    def printNotes(self):
        """
            Print notes directly to stdout
        """
        for i in self.messages:
            print(i)

    def logNotes(self) -> str:
        """
            Get the compiled note list
            Returns:
                notes: str with line terminating characters
        """
        notes = 'Start messages here\n'
        for i in self.messages :
            notes = notes + '\t' + i + '\n'
        notes = notes + 'End messages here\n'
        return notes
