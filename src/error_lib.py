class FileError(IOError):

    def __init__(self, message=None, file=None, errtype=None):
        self.err_message = ""
        if errtype:
            self.err_message += f"[ErrType: {errtype}]"
        if file:
            self.err_message += f" [File: {file}]"
        if message:
            self.err_message += f" {message}"
        super().__init__(self.err_message)

