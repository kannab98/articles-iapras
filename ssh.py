import paramiko
import os

class Client():
    def __init__(self, hostname, **kwargs):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if kwargs:
            self.ssh.connect(hostname, username='ponur', key_filename= os.path.join(os.path.expanduser("~"), '.ssh', 'id_rsa'))
        else:
            self.ssh.connect(hostname, **kwargs)


client = Client('10.62.4.173', username='ponur')
stdin, stdout, stderr = client.ssh.exec_command('ls')
for line in stdout:
    print('... ' + line.strip('\n'))
client.ssh.close()
