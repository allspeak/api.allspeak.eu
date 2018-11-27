from flask import jsonify

class RequestException(Exception):

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message

    def __str__(self):
        return self.message


class RequestExceptionPlus(Exception):

    def __init__(self, message1, message2):
        Exception.__init__(self)
        self.message1 = message1
        self.message2 = message2

    def get_msg1(self):
        return self.message1

    def get_msg2(self):
        return self.message2