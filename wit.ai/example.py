from wit import Wit
import os

client = Wit(os.environ['WIT_TOKEN'])

resp = client.message('what is your name?')
print('Response: {}'.format(resp))


