# PKI

## Concepts

* PKI: Public Key Infrastructure

* TLS: Transport Layer Security
* SSL: Secure Sockets Layer

* RSA: Rivest–Shamir–Adleman


* CSR: Certificate Signing Request
* CA: certificate authority
* PEM: Privacy Enhanced Mail

## Encryption and Decryption

Symmetric encryption and decryption

```bash
echo "private message" > plaintext.txt

KEY=`openssl rand -hex 32`
IV=`openssl rand -hex 16`

openssl enc -aes-256-cbc -K $KEY -iv $IV -in plaintext.txt -out encrypted.txt
openssl enc -aes-256-cbc -d -K $KEY -iv $IV -in encrypted.txt -out decrypted.txt
```


## Key

### private key

Generate private key
```bash
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out private_key.pem
```

```
-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC/HvhWEGsLHE9a...
-----END PRIVATE KEY-----
```

* PEM encoding
* Base64 encoding

Generate RSA private key
```
openssl genrsa -out sa.key 2048
```

### public key

```
openssl rsa -pubout -in private_key.pem -out public_key.pem
```

## Verification

Sign with private key, verify with public key.

```
openssl dgst -sha256 -sign private_key.pem -out signature message.txt
openssl dgst -sha256 -verify public_key.pem -signature signature message.txt
```


## Encryption and decryption

Encrypt with public key, decrypt with private key.

Example with `genkey`

```
openssl genpkey -algorithm RSA -out private_key.pem -pkeyopt rsa_keygen_bits:2048
openssl rsa -pubout -in private_key.pem -out public_key.pem

openssl rsautl -encrypt -inkey public_key.pem -pubin -in message.txt -out message.enc
openssl rsautl -decrypt -inkey private_key.pem -in message.enc -out message.dec
```

Example with `genrsa`
```
openssl genrsa -out sa.key 2048
openssl rsa -pubout -in sa.key -out sa.pub

openssl rsautl -encrypt -inkey sa.pub -pubin -in message.txt -out message.enc
openssl rsautl -decrypt -inkey sa.key -in message.enc -out message.dec
```

## x509

CA

```shell
openssl genpkey -algorithm RSA -out ca.key
openssl req -x509 -new -nodes -key ca.key -sha256 -days 365 -out ca.crt -subj "/C=CN/ST=Beijing/L=Beijing/O=Company/OU=IT/CN=CA"
```

Server
```shell
openssl genpkey -algorithm RSA -out server.key
openssl req -new -key server.key -out server.csr -subj "/C=CN/ST=Beijing/L=Beijing/O=Company/OU=IT/CN=localhost" -addext "subjectAltName=DNS:localhost"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -sha256 -extfile <(printf "subjectAltName=DNS:localhost")
```

Client
```shell
openssl genpkey -algorithm RSA -out client.key
openssl req -new -key client.key -out client.csr -subj "/C=CN/ST=Beijing/L=Beijing/O=Company/OU=IT/CN=localhost" -addext "subjectAltName=DNS:localhost"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 365 -sha256 -extfile <(printf "subjectAltName=DNS:localhost")
```

Show the cerfificate
```shell
openssl x509 -text -noout -in server.crt
openssl x509 -text -noout -in client.crt
```

```
curl -k https://localhost:8443
```

```
curl --cacert ca.crt https://localhost:8443
```

```
curl --cacert ca.crt --key client.key --cert client.crt https://localhost:8443
```

| client \ server | CA | SelfSigned | VerifyClient | --- |
| --- | --- | --- | --- | --- |
| CA | OK | - | - | |
| skip | OK | OK | NO | |
| client | - | - | OK | |

Munual Verification

```
openssl s_client -connect localhost:8443 -showcerts
```

```
openssl verify -CAfile ca.crt server.crt
```

## ssh-keygen

TODO: Can ssh-key pair used to verification ?

```shell
# Generate private key
ssh-keygen -t rsa -b 2048 -f id_rsa

# Extract public key
ssh-keygen -y -f id_rsa > id_rsa.pub

# Convert public key to pem format
ssh-keygen -f id_rsa.pub -e -m pem > id_rsa_pub.pem

# Sign file with private key
openssl dgst -sha256 -sign id_rsa -out signature.sig id_rsa.pub

# Verify the signature with public key
openssl dgst -sha256 -verify id_rsa_pub.pem -signature signature.sig id_rsa.pub
```

```
openssl rsa -in id_rsa -pubout -out id_rsa.pub.pem
```

```
ssh-keygen
```

| Command | private | public | --- | --- |
| --- | --- | --- | --- | --- |
| ssh-keygen | id_rsa | id_rsa.pub | --- | --- |
| openssl rsa | sa.key | sa.pub | --- | --- |


## References

* https://en.wikipedia.org/wiki/Public_key_infrastructure
* https://serverfault.com/questions/9708/what-is-a-pem-file-and-how-does-it-differ-from-other-openssl-generated-key-file
* https://www.ssh.com/academy/ssh/keygen
