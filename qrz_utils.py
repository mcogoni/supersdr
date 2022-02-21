import os
import requests
import xmltodict

class QRZerror(Exception):
    pass

class CallsignNotFound(Exception):
    pass

class QRZsessionNotFound(Exception):
    pass

class QRZMissingCredentials(Exception):
    pass

class QRZ(object):
    def __init__(self, cfg_filename):
        try:
            with open(cfg_filename, "r") as fd:
                data = fd.readlines()
            for row in data:
                if "username" in row:
                    self.username = row.split("=")[1].rstrip()
                if "password" in row:
                    self.password = row.split("=")[1].rstrip()
        except:
            self.username = None
            self.password = None
        self._session = None
        self._session_key = None

    def _get_session(self):
        if not self.username or not self.password:
            raise QRZMissingCredentials("No Username/Password found")

        url = '''https://xmldata.qrz.com/xml/current/?username={0}&password={1}'''.format(self.username, self.password)
        self._session = requests.Session()
        self._session.verify = bool(os.getenv('SSL_VERIFY', False))
        r = self._session.get(url)
        if r.status_code == 200:
            raw_session = xmltodict.parse(r.content)
            self._session_key = raw_session.get('QRZDatabase').get('Session').get('Key')
            if self._session_key:
                return True
        raise QRZsessionNotFound("Could not get QRZ session")

    def callsign(self, callsign, retry=True):
        if self._session_key is None:
            self._get_session()
        url = """http://xmldata.qrz.com/xml/current/?s={0}&callsign={1}""".format(self._session_key, callsign)
        r = self._session.get(url)
        if r.status_code != 200:
            raise Exception("Error Querying: Response code {}".format(r.status_code))
        raw = xmltodict.parse(r.content).get('QRZDatabase')
        if not raw:
            raise QRZerror('Unexpected API Result')
        if raw['Session'].get('Error'):
            errormsg = raw['Session'].get('Error')
            if 'Session Timeout' in errormsg or 'Invalid session key' in errormsg:
                if retry:
                    self._session_key = None
                    self._session = None
                    return self.callsign(callsign, retry=False)
            elif "not found" in errormsg.lower():
                raise CallsignNotFound(errormsg)
            raise QRZerror(raw['Session'].get('Error'))
        else:
            ham = raw.get('Callsign')
            if ham:
                return ham
        raise Exception("Unhandled Error during Query")

