import base64

def encode(data):
    encoded_bytes = base64.b64encode(data.encode('utf-8'))
    encoded_str = encoded_bytes.decode('utf-8')
    return encoded_str

def decode(data):
    decoded_bytes = base64.b64decode(data)
    decoded_string = decoded_bytes.decode('utf-8')
    return decoded_string

print(encode("AKIA4MI2JSNSQVVGU7QP"))
print(encode("DABRHb8j51TyceoSOWsEAnkNedJHMatgy6O/IUsm"))