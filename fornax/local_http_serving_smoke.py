from __future__ import annotations

import hashlib
import json
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .serving import simulate_serving_adapter, validate_serving_adapter_fixture


RECORD_KIND = "local-http-serving-smoke"
EVIDENCE_SCOPE = "local-http-sse-serving-smoke"
PLAN_ID_HEADER = "x-fornax-plan-id"
PLAN_HASH_HEADER = "x-fornax-plan-hash"
AUTH_HEADER = "authorization"
LOCAL_LIFECYCLE_RESOURCE_KINDS = (
    "request_envelope",
    "engine_context",
    "scheduler_slot",
    "response_stream",
    "kv_cache",
)
BACKEND_MODE_ADAPTER = "adapter"
BACKEND_MODE_TARGET_FIXTURE = "target-fixture"
BACKEND_MODES = (BACKEND_MODE_ADAPTER, BACKEND_MODE_TARGET_FIXTURE)
TARGET_FIXTURE_MODEL_ID = "fornax-local-target-fixture-v1"
TARGET_FIXTURE_TEMPLATE_HASH = "sha256:" + "c" * 64
TARGET_FIXTURE_TOKENIZER_HASH = "sha256:" + "d" * 64
TARGET_FIXTURE_STOP_SEQUENCE = "</final>"
LOCAL_TLS_CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDJTCCAg2gAwIBAgIUVheQ23kqqwZE2L4OnSDAbGEr1kcwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDYyMjE3MjIzNloXDTM2MDYx
OTE3MjIzNlowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAyqaqP6VcPGTlm4ebBkKg3/aEIv5+K6pgLv8nJi5bQZPx
DVoMQmzvzbnOFI+yRdeOkpppLtds3BQlkAxIHDrDp/UZ+Yym8KtkSGwtEq8N5c1P
r8IwRb37fc5PkbOU+sMS2g1bkymq1aRQc8ELYfOvhGsu6K/EZdltium0TBiRXc+m
3Mi/M6FIqPCMJcaaPp2FGUK7AagCLkl4y4E4Wg2+OSn8f4it15ex2kmBLlvbzuAt
tZDT2CZPzaQO/uGwi7IJmdbwlrNJ0e86YrNPOdO668M7opDoCvmUvzXXj3HAv2sc
gNdVD6S9JEl5P3+ZmS4V3oqeAllsAj1bGWKf/b5QsQIDAQABo28wbTAdBgNVHQ4E
FgQU3QCSExc/sltQ9FnxBHHNMpz/5ukwHwYDVR0jBBgwFoAU3QCSExc/sltQ9Fnx
BHHNMpz/5ukwDwYDVR0TAQH/BAUwAwEB/zAaBgNVHREEEzARgglsb2NhbGhvc3SH
BH8AAAEwDQYJKoZIhvcNAQELBQADggEBALQZqFNra7iD5Vj6htmt7eU9H5ABviXG
HpCTJmcvBCG0RKTmWTx4+4Hp1LcUGCRwzO//Ix+4yPUD0jT1pBmXeLXf8BiDnT8W
h5SWOuLelirkFJ5UBRVUsmfW4KZv/yCeDacv15iS2kLvyJsr88Z9AxuhuRzo7FCH
B/BYZRyMq0nAUHOkqSp/Rfbut3PiWCaY78Y+oV7dLdHkjM7lCXG7x9oJ9sp0ziVa
Ef6Qc6AXXZADOlNlMEHwl9hURqrK29MXLs9b8zi/ojlkZyZncgIiaPD6jlimzTBH
e6PpGhd5WjJ8fRinR3w+8L3Dbrkt4uOp2FU/E1LiE2qCmG5zV7P5F8A=
-----END CERTIFICATE-----
"""
LOCAL_TLS_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDKpqo/pVw8ZOWb
h5sGQqDf9oQi/n4rqmAu/ycmLltBk/ENWgxCbO/Nuc4Uj7JF146Smmku12zcFCWQ
DEgcOsOn9Rn5jKbwq2RIbC0Srw3lzU+vwjBFvft9zk+Rs5T6wxLaDVuTKarVpFBz
wQth86+Eay7or8Rl2W2K6bRMGJFdz6bcyL8zoUio8Iwlxpo+nYUZQrsBqAIuSXjL
gThaDb45Kfx/iK3Xl7HaSYEuW9vO4C21kNPYJk/NpA7+4bCLsgmZ1vCWs0nR7zpi
s08507rrwzuikOgK+ZS/NdePccC/axyA11UPpL0kSXk/f5mZLhXeip4CWWwCPVsZ
Yp/9vlCxAgMBAAECggEABXlzS8UGukPwgZ8sGdhxb54Wy4+L7tfgrCGs7VFfGjpU
o/Yc3YDYs+TzKKeJcEJmjIElhRmBNusdaP2H8ilH5qY5UCjo9piZXNfSmjixO6ts
SGogHeDdZcnjE44iIL8QfTmopw+prjFM52pINC/I0Undn8q7F77uqwwYAzPpCLtv
w4hXJziRPegMCm9ASmZPcaMgx79/+K23G2n3387Dn7uGcvJ0HqvR4F5C2utjveqk
2Vgn+FvLEOUgvzL3fRDvCOFfXxTyihv0GE+qntFGJ0VgK7/rgxXjtH2Avaf0k/ZQ
1rRuIadayMKcx2jKEREYKnCmfsvwyUNHUGsFB8AgpQKBgQDpQMSV3hiQFDJPZ0Sj
TCHVWYerTwRyIJf31Amr7m9awvPy5RGKKnAyUAEW7gOj64/MwBJ1kL1UVsYKl2tZ
jiiSAI4fV5Zybbiy1gIlzyrcXSEkHJjXwvPNHvB4vaqFOrPPKd97+J0YK3N+8iW4
dOwabBRk4EXrhYoHJ0XEAzaVVQKBgQDeaejm3r0I9SYRGiQckbtkkLCHk3YjlzdG
y5IEQLmkqgpLU9fOVUJQe3Qiyx+9wQ+DDL/KNS05yINt/4+wk5sFdwcoQNyrkeIU
EsUBfFDin56LqooSCjJe3p+K3/4rlp0JvKFgprPoGPCN3Ky51/h+ECAHD7iukpkM
SvHJ1B3N7QKBgQCGjGHZwV+R3NSYkQ061TO/CgIEg3QhEUQYJSvfDY8WX9awigpw
FMLbguLeAzX+XGd6yGDdiDxuZg+fFHFMG4Czl7ZjxfZ202vzXReoD7S9oMr5NbXE
4CQacnpsa5vtdks6eQD9Vg/oXUgmNjAkEu4O38Fz3xr2HPXd4n2P7/qQcQKBgCWc
zYk1g8xfANgFjrPSJVmlamUTF/h+2xc61++mLn7dTq5ceHNpUbSgnAxCQ5TocEIe
RtTgV0ydTzSr8lXPMHklHu28wlS1cAErB2vv5RHeIobGCWFxngETLvHiXW5roxUB
dF0O8/+9L/kdp4wqLNjMy03GZ9oF6qH8jpUuLPglAoGAQDAwsZOeJiSygUDFGix0
cLSZyAWtXCDVg1hVgSAvLl/P82D8JmX3us2iXPayrhoeCMIyKx6V17at3OTfVVd6
lCbaQkdbIjz4FgKRiHgwc/btZ7jYH++WZHKnHgMBDoFjmx88o/JlXDh7zUwioQjL
d0Oa4T0baB+z47Vdp45xad8=
-----END PRIVATE KEY-----
"""
LOCAL_TLS_CERT_SHA256 = "sha256:" + hashlib.sha256(
    ssl.PEM_cert_to_DER_cert(LOCAL_TLS_CERT_PEM)
).hexdigest()
LOCAL_TLS_SUBJECT_ALT_NAMES = ["DNS:localhost", "IP:127.0.0.1"]
LOCAL_TLS_MINIMUM_VERSION = "TLSv1.2"
# Local smoke certificate fixtures; these are not production secrets.
LOCAL_MTLS_CA_CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDMTCCAhmgAwIBAgIUdUCJmk4IHdizMPw/FH1No6EI3x8wDQYJKoZIhvcNAQEL
BQAwIDEeMBwGA1UEAwwVRm9ybmF4IExvY2FsIFNtb2tlIENBMB4XDTI2MDYyMjE3
Mzg0OFoXDTM2MDYxOTE3Mzg0OFowIDEeMBwGA1UEAwwVRm9ybmF4IExvY2FsIFNt
b2tlIENBMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxE+6gc8gbbW2
5y7dWXHEZR1rZzTYGsA203q5mTwjlmt6VS/EALzjvIRzZla3d3eWtlzLUYLVBNox
qISm8lIMQ3TZaFXP3gYphN5jJJ/ykl0v7v6oH6m36S392w6CwoIrjjWHANtHiS8D
GQkTfXpwPOVdr5liAHu4r3fpX7dYVe/t70KGACYlKnE9GDCXGUvf5l5q3Ce8EIsb
d8UTFx03AtFHKVKHO8CM7EU3agCW77/hqRg9NpuH+QVpghDngARdrImDwFZmJ2tH
Uco1W8EL4Jub5Dkx5LjLnB4p01i2VkkEc9PRV+XR5VubfXfDFrLC18QKeu/YhSxd
WwNLWI2EJwIDAQABo2MwYTAdBgNVHQ4EFgQUeuXWFbVVRDklf1ULpbN5wmIYHB8w
HwYDVR0jBBgwFoAUeuXWFbVVRDklf1ULpbN5wmIYHB8wDwYDVR0TAQH/BAUwAwEB
/zAOBgNVHQ8BAf8EBAMCAQYwDQYJKoZIhvcNAQELBQADggEBAKs5BQD78FfQO5aE
grC1VBwDtfDMNJMLtxFhBN1zGfhW/YDZZQffCaNyibgrCi6rXWCdWmMnWx3MIVyW
dyQvbNJA4lG0FIZsYhcdV9g5pvLAYwQGckxddFa3DR31OarM4qp4JJntmn1z95cl
pu1Qm9hJOVkAL0R/hn1ef7wYyh8Sr0/It/2Qq4RuQT8QQInqE4+TWaEHBHUH1ePm
c2ozBniNDAzhi9gDWfYKZx/m2vO9WRLgBfEGbZkv7U6oJNgFry4Hv9aD/Kmf2zLn
2ZG92DpONIdetCOY34ZU5wJtowT3yS2IHnxLENYAhaHaQkUiNuH8W3FFE4r2Livw
acX2g5I=
-----END CERTIFICATE-----
"""
LOCAL_MTLS_SERVER_CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDQzCCAiugAwIBAgIUchuVfNc2Hb7K9k3xH8/zzsUaK8wwDQYJKoZIhvcNAQEL
BQAwIDEeMBwGA1UEAwwVRm9ybmF4IExvY2FsIFNtb2tlIENBMB4XDTI2MDYyMjE3
Mzg0OFoXDTM2MDYxOTE3Mzg0OFowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjAN
BgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAuVaGmuLJ54UkDYPA7ZdQT3TjxXWH
kDglB7VBK4nML9nHH2Rtfa5UHcpx/b4WaMRYap3pkTO7E65iEXxleTZfSxjm3xQC
yPcT4FVSpYyiic/0Jk5dKejnwZbSJcVyZ9Pudm/ncxqkMjFQaZlTNSa9lXWzh00J
vkOV1Ktj/nUh8inaz7yE/4E+0cNJajAVFHGg5FVCyki1NAD1GTGTOVxkK17mCf38
4uesSyGxPJxKzwnT6uSdBNj8BRDhv8R3tHk4IHzt94qzAIe4oxCgCDpPX2f9HQ8s
hgdQyEOGJg12J4fdrJ1Laj9WDDWU0mpDfOsK09lZztmbrWP8qFCXE2DncQIDAQAB
o4GAMH4wGgYDVR0RBBMwEYIJbG9jYWxob3N0hwR/AAABMBMGA1UdJQQMMAoGCCsG
AQUFBwMBMAsGA1UdDwQEAwIFoDAdBgNVHQ4EFgQUNRtzdIsMyeQncAtUoTRYzmNA
5O4wHwYDVR0jBBgwFoAUeuXWFbVVRDklf1ULpbN5wmIYHB8wDQYJKoZIhvcNAQEL
BQADggEBAAgxOs+IkId+IKgnguSnX8fTlWiwJTtl52pj0nnUFBHsrsqbLPKpMU9V
MBgDGvlJEUspDD4mRofadWZ8KMQgJFZytD8qAh5xMonUcritLPc37wmfbH/wwjbB
EYaf/WWFVhUaq/gPfX3GGImdnndvrQGyuJgzCC8DCRDwNLNkOWP1C6c2tJr++8In
MEMyW6ewfxu9dX/8Ce9ASjjSuN8WYMKWDe8JnmRFJ0WIflSGyW14YMRLj1fdypZT
FZUelTaDwGi2zd2Rimhio6Zd31rFh9S8V3g9XKMhgYeS2vzBuDf8SiMyU95CwXdM
0blTlwvG7uKo+dfT6M8B0xeQjUUlWow=
-----END CERTIFICATE-----
"""
LOCAL_MTLS_SERVER_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC5Voaa4snnhSQN
g8Dtl1BPdOPFdYeQOCUHtUEricwv2ccfZG19rlQdynH9vhZoxFhqnemRM7sTrmIR
fGV5Nl9LGObfFALI9xPgVVKljKKJz/QmTl0p6OfBltIlxXJn0+52b+dzGqQyMVBp
mVM1Jr2VdbOHTQm+Q5XUq2P+dSHyKdrPvIT/gT7Rw0lqMBUUcaDkVULKSLU0APUZ
MZM5XGQrXuYJ/fzi56xLIbE8nErPCdPq5J0E2PwFEOG/xHe0eTggfO33irMAh7ij
EKAIOk9fZ/0dDyyGB1DIQ4YmDXYnh92snUtqP1YMNZTSakN86wrT2VnO2ZutY/yo
UJcTYOdxAgMBAAECggEAH9SX7o7nnS8VF4savdNgdBz/p2B9hUvXNEExbaBPZKJy
29XcJhskEC++LxWP9LNP+hOf0xO+2+z1o8opBU3MxPGEaSifJ9OjYsJUnOEP4zpH
VLLdtj5tWyajdeA4u1M/H7EtX2viakpk5JX0H+tE4R/jUufstfQWjf1XgZKtX1D8
yZIoicnFg0Gp9ghctu6S4SfMixKdQCUPfrSII03QiZLPEtEFMNut98RZnPOJKBa7
bNTgHQt38fxa2RXGzohLua1y5FdssHyvvjD22exYQfJxw4duD9dJCE7DQmiS9mgx
rwPLzQ/Tat51pOXPXTVL1t5TbznEFYodY0cHCIAIbwKBgQDuCePMADLiy2EO5GIS
YGkScrDJroyvsIclPlWigWdeC2nW3j4Khe8L2mPcZ0RnitwyXnuHaHNCUBvknu50
cDwOpKp2tGdkJQyID244SPoLzb4eNWQCjU2amf7v1czZe4iOFMOgIpriYd/7p641
xiLQZxQ0R8IsmyRA1pbmMD4cHwKBgQDHUqLIH6FY2fifCYh8j65m2h1vUdzVm5+T
baPGyBWBr8BHB0P9SIP5dp0dwa1nhbrxITtbpJJ2xWINWK6fMgSuhyIu/Dasmb3B
6pq6YVLL6k4u3//T+uaoU8c0+YsDEVfLy7grsrtPeXtsCKoaLXKVPVsGiYgxmPWw
2PXDaXaKbwKBgQCpb163YdjWuD+Q8x2fyQJhgEO4e0dm8zdvWixobMXgGi2vJShy
Ix2hiUUVqGf3b88HB1vUaZPJOu5v0HUZap5FKg6wSf9iOEDwRFOHOuLJLhzKseRL
MLtxdXYSz1Nt5tGvmLYasScgRtzA631Eh5FWWFj4Ua+0QoYOpPBqTyKWrwKBgDWI
AwHeRNX5DGWiCM7zQ7KJx7f682VG+2971a1wVR5UVj59PAxNKmyYJ5AzUN/psZBL
DYcKRu+xCSludM96fnlk/5BA2mo2jBkFafK+ap8rWT4LmMiUrNfuUCTxFB2JzduE
5fyObvHkPXqBRTXsmMZuCQFTdIllfC63xiFqNJTvAoGBAN/p1ismxRLcp85d17N3
59vlOTG5pJ71dvlX2C5lRFrb5gc54rYMAOunWIxbC0PyjBVVAUAGWQuHiDBu57tl
JA/ZN19X7h8kc7estb3VS4CboWKWx1bRgROXkWH+5HICmDYZLUeLvVEC3GZf6tv1
lfGmbwIjxjTHLzdwL+cB0vmA
-----END PRIVATE KEY-----
"""
LOCAL_MTLS_CLIENT_CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDMDCCAhigAwIBAgIUchuVfNc2Hb7K9k3xH8/zzsUaK80wDQYJKoZIhvcNAQEL
BQAwIDEeMBwGA1UEAwwVRm9ybmF4IExvY2FsIFNtb2tlIENBMB4XDTI2MDYyMjE3
Mzg0OVoXDTM2MDYxOTE3Mzg0OVowHjEcMBoGA1UEAwwTZm9ybmF4LWxvY2FsLWNs
aWVudDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALxMCjL+t6M7ezVa
GIfeJgrZt5+ZsE3NRJHbTl00C3gne0OZI+s6vUViG8bjJ+hIMApdILm9tNZIxKOI
ppuQl0jrXydEjoc3LQi9cJqCxGHqE2Ww94dQYuBSrnnlVOr/hFb8tU0wsPMFRz/C
T08o9bG0yCGulK8bbA/lrhHw0NECrln/rA2jrjjMZpX/1aDIKkcSkRdjeowAPxoM
Q3pUp3PAcl3GCof5t37y2Hk66/b32bcba6C9+0cwwal4u4fUmwnra0pL6DDfO74G
PMeGpzUrjCE3t3pPiziFH9d+x/FMojOz1D7brf5EYtwNQlrr/CG8eWXYX68+rY4b
l78StrcCAwEAAaNkMGIwEwYDVR0lBAwwCgYIKwYBBQUHAwIwCwYDVR0PBAQDAgeA
MB0GA1UdDgQWBBRirN0omrnkEjFUl/IsDM1IZPn5uDAfBgNVHSMEGDAWgBR65dYV
tVVEOSV/VQuls3nCYhgcHzANBgkqhkiG9w0BAQsFAAOCAQEAFX5T5g/SQ5VC4UPY
L+v9bx3WIeIie0F8jcMwwYgwn5mw6R74RGmK3Fc+efHpWtOUI70EGgY5uwWxWGXe
uBl5bwEWzmAAJmU1Xy7vEAMs0n9lMebllOgvgwC4aPNXjW+IsLVCVDg276u700vh
LH0S6PZLIA5Wz0hFtZKL7JENqmM+FY9VH4j5Fsy+t2PT60udSSy7PfY9bJXHlya4
uNnZxxAZYXHlggk513oJeUcSk7quW99W8rx+1u5jA/XKIlJNIfxziii0K1ZzyE65
ES/L7HRgAHSzUEWXtML7omzO7TXkQa4Ci85zT3yn6uYxiH14wxeNG3v6GVCkhVVc
KqyLMQ==
-----END CERTIFICATE-----
"""
LOCAL_MTLS_CLIENT_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC8TAoy/rejO3s1
WhiH3iYK2befmbBNzUSR205dNAt4J3tDmSPrOr1FYhvG4yfoSDAKXSC5vbTWSMSj
iKabkJdI618nRI6HNy0IvXCagsRh6hNlsPeHUGLgUq555VTq/4RW/LVNMLDzBUc/
wk9PKPWxtMghrpSvG2wP5a4R8NDRAq5Z/6wNo644zGaV/9WgyCpHEpEXY3qMAD8a
DEN6VKdzwHJdxgqH+bd+8th5Ouv299m3G2ugvftHMMGpeLuH1JsJ62tKS+gw3zu+
BjzHhqc1K4whN7d6T4s4hR/XfsfxTKIzs9Q+263+RGLcDUJa6/whvHll2F+vPq2O
G5e/Era3AgMBAAECggEAV8zhKJwrRtryfZMwQxJsDL0uaSZB1lVztstbBAzDmFhY
mtGqtQHjxZmUuC2tqxsWA48fNlzmGNE/l72daoaGdIMEEIxgJV6uChhDjFiTd9Ct
EMru8NKj+FO1dbNg4F8a93DInWCp8fexHGLfllrUDfPtKf985xXTUqpXe5gd1ocq
tg/TYWivBnZQSQDCDc4CYmgSNZ4D505gWh0DgpkfCfJbFFHYlGwvAzvIVD5KYK97
hBJ4ztx/eQb93NthdNh4+NLY4AeyHVvwXVCRXoONcO0OITPg4dwiumxbBsz6rgiT
7w2RsxpKMKB1TOoNYetAcaVvt4QD6iF5rFru29COIQKBgQDy7ZLEzXnI7aFAvweT
cTHq8GA4aHljXzEpoxLd6D3O1cXm6iPCToUC5YtDQuGDeOkmGQ9sS6fJXKHD4WOT
XbE8aq52mxTQY36p1lu6dxeQVQtiPl+cQqkgIAwYd9b3z7rNP+01x4yh2U3jFBKF
NJX9k3sbnAq2/LzXzi94GZsmhwKBgQDGbedXYMRvcQF018HBeSnlB09l0CDW/Gck
Vq9tzORkojj/L5pd0ZannpEUOClzoD8Ays552L/KE3PMu+8BG6eOOjXLuIR9Myso
s5OzUgJv3OVwT7b09fJ+2C1Av3/bogpylNli1vGDXI6jhhKg1b3YC9999zXRZIdQ
uHhMUwPKUQKBgHzhOCPHZSWvUsfP6/sVo42cyDn3Kv+0fbdjx10f+DYNmtCb6IoI
h0P38GFBTmChlWkqVM1dDwHqhpYFlYS6E1R1mv4JtudxXjm8oib29bwSm+mDGu9f
LUYAc3dYk7+MoADHLhAJZvgEl492UBb982UJna2Rx8hNoF5n9esNbr6dAoGAXZHF
3XZMKyDmxupW+5zfHJjt39zdH4O2P3SBFQ3hRXMZ3XvdFxCWMkSbtSUmpteR3hXE
d8C179xsZsbYVXVs9ayNYZuJHmDaoT7ND2pEq+tGZkewxqKTzyyxai7jY/ZtZsq6
F9mz2XXz7Thz4FPqSc6PfR7tyefVx0K1t4gN6YECgYEAnV5b2YUT9r5dnDK7zvpi
kvur2HUxrlDU6Fq2Z6z1vQYbv489RI0plt6JrWq6CWdR9sYvyfyPY3MWrJbmafT2
WD0wBMqt5FX98vP3boXc8tB6BGJ65M+p645uSAl8K4Ai0knL63u3tvD4J7ohP6ti
Yw8oleYHeTIoG2F+VmYvCig=
-----END PRIVATE KEY-----
"""
LOCAL_MTLS_CA_CERT_SHA256 = "sha256:" + hashlib.sha256(
    ssl.PEM_cert_to_DER_cert(LOCAL_MTLS_CA_CERT_PEM)
).hexdigest()
LOCAL_MTLS_CLIENT_CERT_SHA256 = "sha256:" + hashlib.sha256(
    ssl.PEM_cert_to_DER_cert(LOCAL_MTLS_CLIENT_CERT_PEM)
).hexdigest()
LOCAL_MTLS_CLIENT_SUBJECT = "fornax-local-client"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


def _messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and isinstance(content, str):
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _fixture_tokenize(text: str) -> list[str]:
    normalized = text.lower()
    for marker in ("\n", "\t", ".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}"):
        normalized = normalized.replace(marker, " ")
    return [token for token in normalized.split(" ") if token]


class LocalFornaxBackend:
    """Local smoke implementation of the FornaxBackend Engine seam."""

    def __init__(
        self,
        *,
        plan_id: str,
        request_id: str,
        model: str,
        max_tokens: int,
        backend_mode: str = BACKEND_MODE_ADAPTER,
    ) -> None:
        if backend_mode not in BACKEND_MODES:
            raise ValueError(f"backend_mode must be one of {BACKEND_MODES}")
        self.plan_id = plan_id
        self.request_id = request_id
        self.model = model
        self.max_tokens = max_tokens
        self.backend_mode = backend_mode
        self.call_count = 0
        self._target_fixture_runs: list[dict[str, Any]] = []

    def complete(self, request: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        self.call_count += 1
        model = str(request.get("model", self.model))
        max_tokens = int(request.get("max_tokens", self.max_tokens))
        if self.backend_mode == BACKEND_MODE_TARGET_FIXTURE:
            return self._complete_target_fixture(request, model=model, max_tokens=max_tokens, stream=stream)
        return simulate_serving_adapter(
            plan_id=self.plan_id,
            request_id=self.request_id,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
        )

    def _complete_target_fixture(
        self,
        request: dict[str, Any],
        *,
        model: str,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        messages = request.get("messages", [])
        prompt_text = _messages_to_text(messages)
        prompt_tokens = _fixture_tokenize(prompt_text)
        stop = request.get("stop", [TARGET_FIXTURE_STOP_SEQUENCE])
        if isinstance(stop, str):
            stop_sequences = [stop]
        elif isinstance(stop, list):
            stop_sequences = [item for item in stop if isinstance(item, str)]
        else:
            stop_sequences = [TARGET_FIXTURE_STOP_SEQUENCE]
        if not stop_sequences:
            stop_sequences = [TARGET_FIXTURE_STOP_SEQUENCE]

        candidate_tokens = ["fixture", "target", "parity", TARGET_FIXTURE_STOP_SEQUENCE, "ignored"]
        generated_tokens: list[str] = []
        finish_reason = "length"
        stop_sequence: str | None = None
        for token in candidate_tokens:
            if len(generated_tokens) >= max_tokens:
                finish_reason = "length"
                break
            if token in stop_sequences:
                finish_reason = "stop"
                stop_sequence = token
                break
            generated_tokens.append(token)
        if stop_sequence is None and len(generated_tokens) < max_tokens:
            finish_reason = "stop"

        generated_token_ids = [3001 + index for index, _ in enumerate(generated_tokens)]
        generated_text = " ".join(generated_tokens)
        prompt_token_count = len(prompt_tokens)
        completion_token_count = len(generated_tokens)
        total_token_count = prompt_token_count + completion_token_count
        engine_events = [
            {"kind": "start", "request_id": self.request_id, "plan_id": self.plan_id},
            *[
                {
                    "kind": "token",
                    "request_id": self.request_id,
                    "plan_id": self.plan_id,
                    "token_id": token_id,
                    "token_text": token,
                }
                for token_id, token in zip(generated_token_ids, generated_tokens)
            ],
            {
                "kind": "finish",
                "request_id": self.request_id,
                "plan_id": self.plan_id,
                "finish_reason": finish_reason,
                "stop_sequence": stop_sequence,
            },
        ]
        openai_chunks: list[dict[str, Any]] = []
        for index, event in enumerate(engine_events):
            chunk: dict[str, Any] = {
                "id": f"chatcmpl-{self.request_id}",
                "object": "chat.completion.chunk",
                "model": model,
                "index": index,
                "engine_event_kind": event["kind"],
            }
            if event["kind"] == "start":
                chunk["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            elif event["kind"] == "token":
                chunk["choices"] = [{"index": 0, "delta": {"content": event["token_text"]}, "finish_reason": None}]
            else:
                chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
            openai_chunks.append(chunk)

        usage = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": total_token_count,
        }
        engine_request = {
            "request_id": self.request_id,
            "plan_id": self.plan_id,
            "messages": messages if isinstance(messages, list) else [],
            "max_tokens": max_tokens,
            "stream": stream,
            "template": {
                "name": "fornax-local-target-fixture-chat",
                "version": "phase3-local-fixture-v1",
                "hash": TARGET_FIXTURE_TEMPLATE_HASH,
            },
            "tokenizer": {
                "name": "fornax-local-target-fixture-tokenizer",
                "version": "phase3-local-fixture-v1",
                "hash": TARGET_FIXTURE_TOKENIZER_HASH,
            },
        }
        engine_result = {
            "request_id": self.request_id,
            "finish_reason": finish_reason,
            "content": generated_text,
            "usage": usage,
            "template_hash": TARGET_FIXTURE_TEMPLATE_HASH,
            "tokenizer_hash": TARGET_FIXTURE_TOKENIZER_HASH,
            "stop_sequence": stop_sequence,
        }
        openai_response = {
            "id": f"chatcmpl-{self.request_id}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        run = {
            "stream": stream,
            "model": model,
            "template_hash": TARGET_FIXTURE_TEMPLATE_HASH,
            "tokenizer_hash": TARGET_FIXTURE_TOKENIZER_HASH,
            "prompt_token_count": prompt_token_count,
            "completion_token_count": completion_token_count,
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "generated_token_ids": generated_token_ids,
            "finish_reason": finish_reason,
            "stop_sequence": stop_sequence,
            "stream_chunk_count": len(openai_chunks),
        }
        self._target_fixture_runs.append(run)
        return {
            "version": 1,
            "record_kind": "local-target-fixture-serving-result",
            "mode": "local-http-target-fixture-smoke",
            "plan_id": self.plan_id,
            "openai_request": request,
            "engine_request": engine_request,
            "engine_result": engine_result,
            "engine_stream_events": engine_events,
            "openai_response": openai_response,
            "openai_stream_chunks": openai_chunks,
            "target_fixture_run": run,
        }

    def _target_fixture_summary(self) -> dict[str, Any] | None:
        if self.backend_mode != BACKEND_MODE_TARGET_FIXTURE:
            return None
        streams = [run for run in self._target_fixture_runs if run["stream"] is True]
        non_streams = [run for run in self._target_fixture_runs if run["stream"] is False]
        first = self._target_fixture_runs[0] if self._target_fixture_runs else {}
        expected_tokens = first.get("generated_tokens")
        parity = (
            bool(streams)
            and bool(non_streams)
            and all(run.get("generated_tokens") == expected_tokens for run in self._target_fixture_runs)
            and all(run.get("template_hash") == TARGET_FIXTURE_TEMPLATE_HASH for run in self._target_fixture_runs)
            and all(run.get("tokenizer_hash") == TARGET_FIXTURE_TOKENIZER_HASH for run in self._target_fixture_runs)
        )
        return {
            "scope": "local-fixture-only",
            "fixture_model_id": TARGET_FIXTURE_MODEL_ID,
            "requested_model": self.model,
            "loaded": True,
            "parity": parity,
            "non_stream_matches_stream": parity,
            "run_count": len(self._target_fixture_runs),
            "stream_run_count": len(streams),
            "non_stream_run_count": len(non_streams),
            "generated_text": first.get("generated_text"),
            "generated_tokens": first.get("generated_tokens", []),
            "generated_token_ids": first.get("generated_token_ids", []),
            "prompt_token_count": first.get("prompt_token_count"),
            "completion_token_count": first.get("completion_token_count"),
            "stream_chunk_count": first.get("stream_chunk_count"),
            "finish_reason": first.get("finish_reason"),
            "stop_sequence": first.get("stop_sequence"),
            "template_hash": TARGET_FIXTURE_TEMPLATE_HASH,
            "tokenizer_hash": TARGET_FIXTURE_TOKENIZER_HASH,
            "real_frontier_model_loaded": False,
            "real_frontier_model_parity": False,
        }

    def summary(self) -> dict[str, Any]:
        if self.backend_mode == BACKEND_MODE_TARGET_FIXTURE:
            fixture = self._target_fixture_summary() or {}
            return {
                "backend": "FornaxBackend",
                "mode": "local-http-target-fixture-smoke",
                "engine_trait_compatible": True,
                "request_count": self.call_count,
                "target_model_loaded": fixture.get("loaded") is True,
                "target_model_scope": "local-fixture-only",
                "target_model_parity": fixture.get("parity") is True,
                "template_hash": TARGET_FIXTURE_TEMPLATE_HASH,
                "tokenizer_hash": TARGET_FIXTURE_TOKENIZER_HASH,
                "stream_chunk_count": fixture.get("stream_chunk_count"),
                "real_frontier_model_loaded": False,
                "real_frontier_model_parity": False,
                "target_fixture": fixture,
            }
        return {
            "backend": "FornaxBackend",
            "mode": "local-http-smoke",
            "engine_trait_compatible": True,
            "request_count": self.call_count,
            "target_model_loaded": False,
            "target_model_scope": "none",
            "target_model_parity": False,
            "real_frontier_model_loaded": False,
            "real_frontier_model_parity": False,
        }


class _SmokeServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], config: dict[str, Any]) -> None:
        super().__init__(server_address, _SmokeHandler)
        self.config = config
        self.backend = LocalFornaxBackend(
            plan_id=config["plan_id"],
            request_id=config["request_id"],
            model=config["model"],
            max_tokens=config["max_tokens"],
            backend_mode=config["backend_mode"],
        )
        self._inflight_lock = threading.Lock()
        self._inflight_count = 0
        self.max_observed_inflight = 0
        self.backpressure_reject_count = 0
        self.inflight_cleanup_count = 0
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_sequence = 0
        self._lifecycle_active_resources: set[tuple[str, str]] = set()
        self._lifecycle_accepted: set[str] = set()
        self._lifecycle_cleanup_count = 0
        self._lifecycle_rejected_count = 0
        self.lifecycle_events: list[dict[str, Any]] = []
        self._mtls_lock = threading.Lock()
        self._mtls_peer_subjects: list[str] = []

    def record_mtls_peer(self, peer_cert: dict[str, Any] | None) -> None:
        subject = ""
        if isinstance(peer_cert, dict):
            for rdn in peer_cert.get("subject", ()):  # tuple-of-tuples from ssl.
                for key, value in rdn:
                    if key == "commonName" and isinstance(value, str):
                        subject = value
                        break
                if subject:
                    break
        with self._mtls_lock:
            self._mtls_peer_subjects.append(subject or "<unknown>")

    def mtls_summary(self) -> dict[str, Any]:
        with self._mtls_lock:
            subjects = list(self._mtls_peer_subjects)
        return {
            "enabled": bool(self.config.get("enable_mtls")),
            "verified_peer_count": len(subjects),
            "peer_subjects": subjects,
            "expected_client_subject": LOCAL_MTLS_CLIENT_SUBJECT,
            "all_peers_expected": bool(subjects)
            and all(subject == LOCAL_MTLS_CLIENT_SUBJECT for subject in subjects),
        }

    def try_admit(self) -> bool:
        with self._inflight_lock:
            if self._inflight_count >= int(self.config["max_inflight"]):
                self.backpressure_reject_count += 1
                return False
            self._inflight_count += 1
            self.max_observed_inflight = max(self.max_observed_inflight, self._inflight_count)
            return True

    def release_inflight(self) -> None:
        with self._inflight_lock:
            if self._inflight_count > 0:
                self._inflight_count -= 1
            self.inflight_cleanup_count += 1

    def wait_for_inflight(self, count: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._inflight_lock:
                if self._inflight_count >= count:
                    return True
            time.sleep(0.005)
        return False

    def backpressure_summary(self) -> dict[str, Any]:
        with self._inflight_lock:
            current_inflight = self._inflight_count
        return {
            "max_inflight": self.config["max_inflight"],
            "current_inflight": current_inflight,
            "max_observed_inflight": self.max_observed_inflight,
            "backpressure_reject_count": self.backpressure_reject_count,
            "inflight_cleanup_count": self.inflight_cleanup_count,
        }

    def _append_lifecycle_event(
        self,
        *,
        kind: str,
        request_label: str,
        resource_kind: str | None,
        owner: str,
        state: str,
        reason: str,
    ) -> None:
        self.lifecycle_events.append(
            {
                "index": len(self.lifecycle_events),
                "kind": kind,
                "request_label": request_label,
                "resource_kind": resource_kind,
                "owner": owner,
                "state": state,
                "reason": reason,
            }
        )

    def record_rejection(self, reason: str) -> None:
        with self._lifecycle_lock:
            self._lifecycle_rejected_count += 1
            self._append_lifecycle_event(
                kind="request_rejected",
                request_label=f"rejected-{self._lifecycle_rejected_count}",
                resource_kind=None,
                owner="serving_gateway",
                state="rejected",
                reason=reason,
            )

    def allocate_lifecycle(self, *, stream: bool) -> str:
        with self._lifecycle_lock:
            self._lifecycle_sequence += 1
            request_label = f"accepted-{self._lifecycle_sequence}"
            self._lifecycle_accepted.add(request_label)
            for resource_kind in LOCAL_LIFECYCLE_RESOURCE_KINDS:
                self._lifecycle_active_resources.add((request_label, resource_kind))
            self._append_lifecycle_event(
                kind="request_received",
                request_label=request_label,
                resource_kind="request_envelope",
                owner="serving_gateway",
                state="active",
                reason="OpenAI-compatible request accepted",
            )
            self._append_lifecycle_event(
                kind="engine_request_normalized",
                request_label=request_label,
                resource_kind="engine_context",
                owner="fornax_engine",
                state="active",
                reason="request normalized into local EngineRequest",
            )
            self._append_lifecycle_event(
                kind="scheduler_admitted",
                request_label=request_label,
                resource_kind="scheduler_slot",
                owner="scheduler",
                state="active",
                reason="local admission slot allocated",
            )
            self._append_lifecycle_event(
                kind="stream_opened" if stream else "response_opened",
                request_label=request_label,
                resource_kind="response_stream",
                owner="serving_gateway",
                state="active",
                reason="serving response state opened",
            )
            self._append_lifecycle_event(
                kind="kv_read_granted",
                request_label=request_label,
                resource_kind="kv_cache",
                owner="kv_manager",
                state="active",
                reason="local smoke KV ownership placeholder opened",
            )
            return request_label

    def release_lifecycle(self, request_label: str) -> None:
        with self._lifecycle_lock:
            released_any = False
            for resource_kind in LOCAL_LIFECYCLE_RESOURCE_KINDS:
                key = (request_label, resource_kind)
                if key in self._lifecycle_active_resources:
                    self._lifecycle_active_resources.remove(key)
                    released_any = True
                    self._append_lifecycle_event(
                        kind="cleanup",
                        request_label=request_label,
                        resource_kind=resource_kind,
                        owner="released",
                        state="released",
                        reason="local endpoint request cleanup",
                    )
            if released_any:
                self._lifecycle_cleanup_count += 1

    def lifecycle_summary(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            event_count = len(self.lifecycle_events)
            active_resource_count = len(self._lifecycle_active_resources)
            accepted_request_count = len(self._lifecycle_accepted)
            rejected_request_count = self._lifecycle_rejected_count
            cleanup_count = self._lifecycle_cleanup_count
            resource_allocated_count = accepted_request_count * len(LOCAL_LIFECYCLE_RESOURCE_KINDS)
            resource_released_count = sum(1 for event in self.lifecycle_events if event["kind"] == "cleanup")
            events = list(self.lifecycle_events)
        return {
            "mode": "local-http-lifecycle-smoke",
            "resource_kinds": list(LOCAL_LIFECYCLE_RESOURCE_KINDS),
            "event_count": event_count,
            "accepted_request_count": accepted_request_count,
            "rejected_request_count": rejected_request_count,
            "cleanup_count": cleanup_count,
            "resource_allocated_count": resource_allocated_count,
            "resource_released_count": resource_released_count,
            "active_resource_count": active_resource_count,
            "all_required_resources_released": active_resource_count == 0,
            "single_owner_preserved": True,
            "events": events,
        }


class _SmokeHandler(BaseHTTPRequestHandler):
    server: "_SmokeServer"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        config = self.server.config
        if config.get("enable_mtls"):
            peer_cert = self.connection.getpeercert() if hasattr(self.connection, "getpeercert") else None
            if not peer_cert:
                self.server.record_rejection("mtls_client_certificate_required")
                self._json_response(
                    403,
                    {
                        "error": {
                            "type": "authentication_error",
                            "code": "mtls_client_certificate_required",
                            "message": "Local smoke mTLS mode requires a verified client certificate.",
                        }
                    },
                )
                return
            self.server.record_mtls_peer(peer_cert)
        if self.path != "/v1/chat/completions":
            self.server.record_rejection("bad_path")
            self._json_response(
                404,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "not_found",
                        "message": "Only /v1/chat/completions is available in local smoke.",
                    }
                },
            )
            return
        expected_auth = f"Bearer {config['auth_token']}"
        if self.headers.get(AUTH_HEADER) != expected_auth:
            self.server.record_rejection("endpoint_auth_required")
            self._json_response(
                401,
                {
                    "error": {
                        "type": "authentication_error",
                        "code": "endpoint_auth_required",
                        "message": "Local smoke endpoint requires the configured bearer token.",
                    }
                },
            )
            return
        plan_id = self.headers.get(PLAN_ID_HEADER)
        plan_hash = self.headers.get(PLAN_HASH_HEADER)
        if plan_id != config["plan_id"] or plan_hash != config["plan_hash"]:
            self.server.record_rejection("plan_integrity_mismatch")
            self._json_response(
                409,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "plan_integrity_mismatch",
                        "message": "Plan identity or hash did not match the local smoke server.",
                    }
                },
            )
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            request = json.loads(body) if body else {}
        except Exception:
            self.server.record_rejection("invalid_json")
            self._json_response(
                400,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                        "message": "Request body must be valid JSON.",
                    }
                },
            )
            return
        stream = bool(request.get("stream", False))
        max_tokens = int(request.get("max_tokens", config["max_tokens"]))
        model = str(request.get("model", config["model"]))
        simulate_work_ms = max(0, int(request.get("simulate_work_ms", 0)))
        if not self.server.try_admit():
            self.server.record_rejection("backpressure_queue_full")
            self._json_response(
                429,
                {
                    "error": {
                        "type": "rate_limit_error",
                        "code": "backpressure_queue_full",
                        "message": "Local smoke inflight capacity is exhausted.",
                    },
                    "retry_after_ms": config["retry_after_ms"],
                },
            )
            return
        request_label = self.server.allocate_lifecycle(stream=stream)
        try:
            if simulate_work_ms:
                time.sleep(simulate_work_ms / 1000.0)
            adapter = self.server.backend.complete(
                {**request, "model": model, "max_tokens": max_tokens},
                stream=stream,
            )
            if stream:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                for chunk in adapter["openai_stream_chunks"]:
                    line = "data: " + json.dumps(chunk) + "\n\n"
                    self.wfile.write(line.encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self._json_response(200, adapter["openai_response"])
        finally:
            self.server.release_lifecycle(request_label)
            self.server.release_inflight()


def _write_local_tls_material(directory: Path) -> tuple[Path, Path]:
    cert_path = directory / "fornax-local-smoke-cert.pem"
    key_path = directory / "fornax-local-smoke-key.pem"
    cert_path.write_text(LOCAL_TLS_CERT_PEM, encoding="utf-8")
    key_path.write_text(LOCAL_TLS_KEY_PEM, encoding="utf-8")
    key_path.chmod(0o600)
    return cert_path, key_path


def _write_local_mtls_material(directory: Path) -> tuple[Path, Path, Path, Path, Path]:
    ca_path = directory / "fornax-local-smoke-ca.pem"
    server_cert_path = directory / "fornax-local-smoke-mtls-server-cert.pem"
    server_key_path = directory / "fornax-local-smoke-mtls-server-key.pem"
    client_cert_path = directory / "fornax-local-smoke-mtls-client-cert.pem"
    client_key_path = directory / "fornax-local-smoke-mtls-client-key.pem"
    ca_path.write_text(LOCAL_MTLS_CA_CERT_PEM, encoding="utf-8")
    server_cert_path.write_text(LOCAL_MTLS_SERVER_CERT_PEM, encoding="utf-8")
    server_key_path.write_text(LOCAL_MTLS_SERVER_KEY_PEM, encoding="utf-8")
    client_cert_path.write_text(LOCAL_MTLS_CLIENT_CERT_PEM, encoding="utf-8")
    client_key_path.write_text(LOCAL_MTLS_CLIENT_KEY_PEM, encoding="utf-8")
    server_key_path.chmod(0o600)
    client_key_path.chmod(0o600)
    return ca_path, server_cert_path, server_key_path, client_cert_path, client_key_path


def _server_tls_context(
    cert_path: Path,
    key_path: Path,
    *,
    client_ca_path: Path | None = None,
    require_client_cert: bool = False,
) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    if hasattr(ssl, "TLSVersion"):
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(cert_path), str(key_path))
    if require_client_cert:
        if client_ca_path is None:
            raise ValueError("client_ca_path is required when require_client_cert is true")
        context.load_verify_locations(cafile=str(client_ca_path))
        context.verify_mode = ssl.CERT_REQUIRED
    return context


def _client_tls_context(
    *,
    ca_pem: str = LOCAL_TLS_CERT_PEM,
    cert_path: Path | None = None,
    key_path: Path | None = None,
) -> ssl.SSLContext:
    context = ssl.create_default_context(cadata=ca_pem)
    if hasattr(ssl, "TLSVersion"):
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    if cert_path is not None and key_path is not None:
        context.load_cert_chain(str(cert_path), str(key_path))
    return context


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    plan_id: str,
    plan_hash: str,
    auth_token: str | None,
    timeout_s: float,
    ssl_context: ssl.SSLContext | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "content-type": "application/json",
        PLAN_ID_HEADER: plan_id,
        PLAN_HASH_HEADER: plan_hash,
    }
    if auth_token is not None:
        headers[AUTH_HEADER] = f"Bearer {auth_token}"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s, context=ssl_context) as response:
            data = response.read().decode("utf-8")
            return {
                "status": int(response.status),
                "content_type": response.headers.get("content-type"),
                "body": json.loads(data),
            }
    except urllib.error.HTTPError as exc:
        data = exc.read().decode("utf-8")
        return {
            "status": int(exc.code),
            "content_type": exc.headers.get("content-type"),
            "body": json.loads(data) if data else {},
        }


def _post_json_transport_error(
    url: str,
    payload: dict[str, Any],
    *,
    plan_id: str,
    plan_hash: str,
    auth_token: str | None,
    timeout_s: float,
    ssl_context: ssl.SSLContext | None = None,
) -> dict[str, Any]:
    try:
        response = _post_json(
            url,
            payload,
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=auth_token,
            timeout_s=timeout_s,
            ssl_context=ssl_context,
        )
    except Exception as exc:
        return {
            "transport_error": True,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "transport_error": False,
        "unexpected_response": response,
    }


def _post_sse(
    url: str,
    payload: dict[str, Any],
    *,
    plan_id: str,
    plan_hash: str,
    auth_token: str | None,
    timeout_s: float,
    ssl_context: ssl.SSLContext | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "content-type": "application/json",
        PLAN_ID_HEADER: plan_id,
        PLAN_HASH_HEADER: plan_hash,
    }
    if auth_token is not None:
        headers[AUTH_HEADER] = f"Bearer {auth_token}"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout_s, context=ssl_context) as response:
        text = response.read().decode("utf-8")
    events: list[dict[str, Any]] = []
    done_seen = False
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if not block.startswith("data: "):
            continue
        raw = block[len("data: ") :]
        if raw == "[DONE]":
            done_seen = True
            continue
        events.append(json.loads(raw))
    return {
        "status": 200,
        "content_type": "text/event-stream",
        "raw_event_count": text.count("data: "),
        "chunk_count": len(events),
        "done_seen": done_seen,
        "events": events,
    }


def run_local_http_serving_smoke(
    *,
    out: str | Path,
    host: str = "127.0.0.1",
    port: int = 0,
    plan_id: str = "local-http-serving-plan",
    plan_hash: str = "sha256:local-http-serving-plan",
    request_id: str = "local-http-serving-request",
    model: str = "qwen3-moe-class-target",
    max_tokens: int = 64,
    auth_token: str = "local-smoke-token",
    max_inflight: int = 1,
    backpressure_delay_ms: int = 250,
    retry_after_ms: int = 25,
    timeout_s: float = 5.0,
    backend_mode: str = BACKEND_MODE_ADAPTER,
    enable_tls: bool = False,
    enable_mtls: bool = False,
) -> dict[str, Any]:
    if not host or not plan_id or not plan_hash or not request_id or not model or not auth_token:
        raise ValueError("host, plan_id, plan_hash, request_id, model, and auth_token must be non-empty")
    if isinstance(port, bool) or not isinstance(port, int) or port < 0:
        raise ValueError("port must be a non-negative integer")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight <= 0:
        raise ValueError("max_inflight must be a positive integer")
    if isinstance(backpressure_delay_ms, bool) or not isinstance(backpressure_delay_ms, int) or backpressure_delay_ms <= 0:
        raise ValueError("backpressure_delay_ms must be a positive integer")
    if isinstance(retry_after_ms, bool) or not isinstance(retry_after_ms, int) or retry_after_ms <= 0:
        raise ValueError("retry_after_ms must be a positive integer")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("timeout_s must be a positive number")
    if backend_mode not in BACKEND_MODES:
        raise ValueError(f"backend_mode must be one of {BACKEND_MODES}")
    if enable_mtls:
        enable_tls = True

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "request_id": request_id,
        "model": model,
        "max_tokens": max_tokens,
        "auth_token": auth_token,
        "max_inflight": max_inflight,
        "backpressure_delay_ms": backpressure_delay_ms,
        "retry_after_ms": retry_after_ms,
        "backend_mode": backend_mode,
        "enable_tls": enable_tls,
        "enable_mtls": enable_mtls,
    }
    server = _SmokeServer((host, port), config)
    server_host, server_port = server.server_address
    tls_tempdir: tempfile.TemporaryDirectory[str] | None = None
    client_ssl_context: ssl.SSLContext | None = None
    mtls_no_client_ssl_context: ssl.SSLContext | None = None
    if enable_tls:
        tls_tempdir = tempfile.TemporaryDirectory(prefix="fornax-local-http-tls-")
        if enable_mtls:
            ca_path, cert_path, key_path, client_cert_path, client_key_path = _write_local_mtls_material(
                Path(tls_tempdir.name)
            )
            server_context = _server_tls_context(
                cert_path,
                key_path,
                client_ca_path=ca_path,
                require_client_cert=True,
            )
            client_ssl_context = _client_tls_context(
                ca_pem=LOCAL_MTLS_CA_CERT_PEM,
                cert_path=client_cert_path,
                key_path=client_key_path,
            )
            mtls_no_client_ssl_context = _client_tls_context(ca_pem=LOCAL_MTLS_CA_CERT_PEM)
        else:
            cert_path, key_path = _write_local_tls_material(Path(tls_tempdir.name))
            server_context = _server_tls_context(cert_path, key_path)
            client_ssl_context = _client_tls_context()
        server.socket = server_context.wrap_socket(
            server.socket,
            server_side=True,
        )
    thread = threading.Thread(target=server.serve_forever, name="fornax-http-smoke", daemon=True)
    started_ns = time.perf_counter_ns()
    thread.start()
    scheme = "https" if enable_tls else "http"
    endpoint = f"{scheme}://{server_host}:{server_port}/v1/chat/completions"
    try:
        adapter = simulate_serving_adapter(
            plan_id=plan_id,
            request_id=request_id,
            model=model,
            stream=True,
            max_tokens=max_tokens,
        )
        adapter_validation = validate_serving_adapter_fixture(adapter)
        non_stream = _post_json(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=auth_token,
            timeout_s=float(timeout_s),
            ssl_context=client_ssl_context,
        )
        stream = _post_sse(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": True},
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=auth_token,
            timeout_s=float(timeout_s),
            ssl_context=client_ssl_context,
        )
        auth_reject = _post_json(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=None,
            timeout_s=float(timeout_s),
            ssl_context=client_ssl_context,
        )
        mtls_reject = (
            _post_json_transport_error(
                endpoint,
                {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
                plan_id=plan_id,
                plan_hash=plan_hash,
                auth_token=auth_token,
                timeout_s=float(timeout_s),
                ssl_context=mtls_no_client_ssl_context,
            )
            if enable_mtls
            else None
        )
        backpressure_holders: list[dict[str, Any]] = []
        backpressure_errors: list[BaseException] = []
        holder_lock = threading.Lock()

        def _hold_inflight_request() -> None:
            try:
                holder = _post_json(
                    endpoint,
                    {
                        "model": model,
                        "messages": adapter["openai_request"]["messages"],
                        "max_tokens": max_tokens,
                        "stream": False,
                        "simulate_work_ms": backpressure_delay_ms,
                    },
                    plan_id=plan_id,
                    plan_hash=plan_hash,
                    auth_token=auth_token,
                    timeout_s=float(timeout_s),
                    ssl_context=client_ssl_context,
                )
                with holder_lock:
                    backpressure_holders.append(holder)
            except BaseException as exc:  # pragma: no cover - re-raised in the main thread.
                backpressure_errors.append(exc)

        holder_threads = [
            threading.Thread(target=_hold_inflight_request, name=f"fornax-http-smoke-hold-{index}")
            for index in range(max_inflight)
        ]
        for holder_thread in holder_threads:
            holder_thread.start()
        inflight_observed = server.wait_for_inflight(max_inflight, float(timeout_s))
        backpressure_reject = _post_json(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=auth_token,
            timeout_s=float(timeout_s),
            ssl_context=client_ssl_context,
        )
        for holder_thread in holder_threads:
            holder_thread.join(timeout=float(timeout_s))
            if holder_thread.is_alive():
                raise TimeoutError("timed out waiting for local backpressure holder request")
        if backpressure_errors:
            raise backpressure_errors[0]
        plan_reject = _post_json(
            endpoint,
            {"model": model, "messages": adapter["openai_request"]["messages"], "max_tokens": max_tokens, "stream": False},
            plan_id=plan_id,
            plan_hash="sha256:mismatch",
            auth_token=auth_token,
            timeout_s=float(timeout_s),
            ssl_context=client_ssl_context,
        )
        bad_path = _post_json(
            f"{scheme}://{server_host}:{server_port}/bad/path",
            {"model": model},
            plan_id=plan_id,
            plan_hash=plan_hash,
            auth_token=auth_token,
            timeout_s=float(timeout_s),
            ssl_context=client_ssl_context,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=float(timeout_s))
        if tls_tempdir is not None:
            tls_tempdir.cleanup()
    backend_summary = server.backend.summary()
    elapsed_ns = time.perf_counter_ns() - started_ns

    non_stream_ok = (
        non_stream.get("status") == 200
        and non_stream.get("body", {}).get("object") == "chat.completion"
        and non_stream.get("body", {}).get("model") == model
    )
    expected_stream_chunk_count = (
        backend_summary.get("stream_chunk_count")
        if backend_summary.get("mode") == "local-http-target-fixture-smoke"
        else adapter_validation["summary"].get("openai_chunk_count")
    )
    stream_ok = (
        stream.get("status") == 200
        and stream.get("done_seen") is True
        and stream.get("chunk_count") == expected_stream_chunk_count
    )
    plan_reject_ok = (
        plan_reject.get("status") == 409
        and plan_reject.get("body", {}).get("error", {}).get("code") == "plan_integrity_mismatch"
    )
    auth_reject_ok = (
        auth_reject.get("status") == 401
        and auth_reject.get("body", {}).get("error", {}).get("code") == "endpoint_auth_required"
    )
    backpressure_summary = server.backpressure_summary()
    lifecycle_summary = server.lifecycle_summary()
    mtls_summary = server.mtls_summary()
    expected_backend_request_count = 2 + max_inflight
    expected_endpoint_request_count = expected_backend_request_count + 4
    backpressure_holder_ok = (
        inflight_observed
        and len(backpressure_holders) == max_inflight
        and all(
            holder.get("status") == 200
            and holder.get("body", {}).get("object") == "chat.completion"
            for holder in backpressure_holders
        )
    )
    backpressure_reject_ok = (
        backpressure_reject.get("status") == 429
        and backpressure_reject.get("body", {}).get("error", {}).get("code") == "backpressure_queue_full"
        and backpressure_summary.get("backpressure_reject_count") == 1
        and backpressure_summary.get("max_observed_inflight") == max_inflight
        and backpressure_summary.get("current_inflight") == 0
    )
    lifecycle_ok = (
        lifecycle_summary.get("accepted_request_count") == expected_backend_request_count
        and lifecycle_summary.get("rejected_request_count") == 4
        and lifecycle_summary.get("cleanup_count") == expected_backend_request_count
        and lifecycle_summary.get("resource_allocated_count") == expected_backend_request_count * len(LOCAL_LIFECYCLE_RESOURCE_KINDS)
        and lifecycle_summary.get("resource_released_count") == lifecycle_summary.get("resource_allocated_count")
        and lifecycle_summary.get("active_resource_count") == 0
        and lifecycle_summary.get("all_required_resources_released") is True
        and lifecycle_summary.get("single_owner_preserved") is True
    )
    bad_path_ok = bad_path.get("status") == 404
    target_fixture_enabled = backend_summary.get("mode") == "local-http-target-fixture-smoke"
    target_fixture_summary = backend_summary.get("target_fixture") if isinstance(backend_summary.get("target_fixture"), dict) else None
    target_fixture_ok = (
        target_fixture_enabled
        and backend_summary.get("target_model_loaded") is True
        and backend_summary.get("target_model_scope") == "local-fixture-only"
        and backend_summary.get("target_model_parity") is True
        and isinstance(target_fixture_summary, dict)
        and target_fixture_summary.get("loaded") is True
        and target_fixture_summary.get("parity") is True
        and target_fixture_summary.get("run_count") == expected_backend_request_count
        and target_fixture_summary.get("stream_run_count", 0) >= 1
        and target_fixture_summary.get("non_stream_run_count", 0) >= 1
        and _is_sha256(target_fixture_summary.get("template_hash"))
        and _is_sha256(target_fixture_summary.get("tokenizer_hash"))
        and target_fixture_summary.get("real_frontier_model_loaded") is False
        and target_fixture_summary.get("real_frontier_model_parity") is False
    )
    backend_target_ok = target_fixture_ok if target_fixture_enabled else (
        backend_summary.get("target_model_loaded") is False
        and backend_summary.get("target_model_parity") is False
    )
    backend_ok = (
        backend_summary.get("backend") == "FornaxBackend"
        and backend_summary.get("engine_trait_compatible") is True
        and backend_summary.get("request_count") == expected_backend_request_count
        and backend_target_ok
    )
    tls_ok = (
        not enable_tls
        or (
            endpoint.startswith("https://")
            and client_ssl_context is not None
            and non_stream_ok
            and stream_ok
        )
    )
    mtls_ok = (
        not enable_mtls
        or (
            isinstance(mtls_reject, dict)
            and mtls_reject.get("transport_error") is True
            and mtls_summary.get("verified_peer_count") == expected_endpoint_request_count
            and mtls_summary.get("all_peers_expected") is True
        )
    )
    checks = [
        {"name": "serving-adapter", "ok": bool(adapter_validation.get("ok")), "errors": adapter_validation.get("errors", []), "warnings": adapter_validation.get("warnings", [])},
        {"name": "fornax-backend-integration", "ok": backend_ok, "errors": [] if backend_ok else ["FornaxBackend local integration invalid"], "warnings": []},
        {"name": "endpoint-auth-reject", "ok": auth_reject_ok, "errors": [] if auth_reject_ok else ["endpoint auth rejection invalid"], "warnings": []},
        *([{"name": "local-tls-handshake", "ok": tls_ok, "errors": [] if tls_ok else ["local TLS handshake invalid"], "warnings": ["local self-signed TLS is not product TLS/mTLS evidence"]}] if enable_tls else []),
        *([{"name": "local-mtls-node-identity", "ok": mtls_ok, "errors": [] if mtls_ok else ["local mTLS node identity invalid"], "warnings": ["local mTLS client identity is not production node identity evidence"]}] if enable_mtls else []),
        {"name": "backpressure-reject", "ok": backpressure_reject_ok and backpressure_holder_ok, "errors": [] if backpressure_reject_ok and backpressure_holder_ok else ["backpressure rejection invalid"], "warnings": []},
        {"name": "lifecycle-cleanup", "ok": lifecycle_ok, "errors": [] if lifecycle_ok else ["lifecycle cleanup invalid"], "warnings": []},
        *([{"name": "target-fixture-parity", "ok": target_fixture_ok, "errors": [] if target_fixture_ok else ["target fixture parity invalid"], "warnings": ["local target fixture parity is not real frontier model parity"]}] if target_fixture_enabled else []),
        {"name": "non-stream-http", "ok": non_stream_ok, "errors": [] if non_stream_ok else ["non-stream HTTP response invalid"], "warnings": []},
        {"name": "stream-sse", "ok": stream_ok, "errors": [] if stream_ok else ["SSE stream response invalid"], "warnings": []},
        {"name": "plan-integrity-reject", "ok": plan_reject_ok, "errors": [] if plan_reject_ok else ["plan integrity rejection invalid"], "warnings": []},
        {"name": "bad-path-reject", "ok": bad_path_ok, "errors": [] if bad_path_ok else ["bad path rejection invalid"], "warnings": []},
    ]
    passed_count = sum(1 for check in checks if check["ok"])
    summary = {
        "check_count": len(checks),
        "passed_count": passed_count,
        "http_endpoint_started": True,
        "endpoint": endpoint,
        "scheme": scheme,
        "host": server_host,
        "port": server_port,
        "non_stream_status": non_stream.get("status"),
        "stream_status": stream.get("status"),
        "sse_chunk_count": stream.get("chunk_count"),
        "sse_done_seen": stream.get("done_seen"),
        "auth_reject_status": auth_reject.get("status"),
        "endpoint_auth_rejected": auth_reject_ok,
        "backpressure_status": backpressure_reject.get("status"),
        "backpressure_rejected": backpressure_reject_ok,
        "backpressure_holder_count": len(backpressure_holders),
        "backpressure_holder_statuses": [holder.get("status") for holder in backpressure_holders],
        "backpressure_holders_completed": backpressure_holder_ok,
        "max_inflight": max_inflight,
        "max_observed_inflight": backpressure_summary.get("max_observed_inflight"),
        "backpressure_reject_count": backpressure_summary.get("backpressure_reject_count"),
        "inflight_cleanup_count": backpressure_summary.get("inflight_cleanup_count"),
        "failure_semantics_verified": backpressure_reject_ok and backpressure_holder_ok,
        "lifecycle_tracked": True,
        "lifecycle_request_count": lifecycle_summary.get("accepted_request_count"),
        "lifecycle_rejected_request_count": lifecycle_summary.get("rejected_request_count"),
        "lifecycle_cleanup_count": lifecycle_summary.get("cleanup_count"),
        "lifecycle_resource_allocated_count": lifecycle_summary.get("resource_allocated_count"),
        "lifecycle_resource_released_count": lifecycle_summary.get("resource_released_count"),
        "lifecycle_active_resource_count": lifecycle_summary.get("active_resource_count"),
        "lifecycle_all_released": lifecycle_summary.get("all_required_resources_released"),
        "lifecycle_single_owner_preserved": lifecycle_summary.get("single_owner_preserved"),
        "plan_integrity_rejected": plan_reject_ok,
        "bad_path_rejected": bad_path_ok,
        "fornax_backend_integrated": backend_ok,
        "backend_request_count": backend_summary.get("request_count"),
        "engine_trait_compatible": backend_summary.get("engine_trait_compatible"),
        "engine_result_emitted": non_stream_ok and stream_ok,
        "backend_target_model_loaded": backend_summary.get("target_model_loaded"),
        "backend_target_model_scope": backend_summary.get("target_model_scope"),
        "backend_target_model_parity": backend_summary.get("target_model_parity"),
        "target_fixture_loaded": bool(target_fixture_enabled and target_fixture_ok),
        "target_fixture_parity": bool(target_fixture_enabled and target_fixture_ok),
        "target_fixture_run_count": target_fixture_summary.get("run_count") if isinstance(target_fixture_summary, dict) else 0,
        "target_fixture_non_stream_matches_stream": target_fixture_summary.get("non_stream_matches_stream") if isinstance(target_fixture_summary, dict) else False,
        "target_fixture_template_hash": target_fixture_summary.get("template_hash") if isinstance(target_fixture_summary, dict) else None,
        "target_fixture_tokenizer_hash": target_fixture_summary.get("tokenizer_hash") if isinstance(target_fixture_summary, dict) else None,
        "real_frontier_model_loaded": False,
        "real_frontier_model_parity": False,
        "elapsed_s": elapsed_ns / 1_000_000_000.0,
        "live_http_endpoint": True,
        "localhost_only": server_host in {"127.0.0.1", "localhost"},
        "local_auth_enabled": True,
        "auth_token_redacted": True,
        "tls_enabled": enable_tls,
        "local_tls_enabled": enable_tls,
        "tls_client_verified": enable_tls,
        "tls_mode": "local-mutual-tls" if enable_mtls else ("local-self-signed" if enable_tls else "disabled"),
        "tls_certificate_sha256": (
            LOCAL_MTLS_CA_CERT_SHA256 if enable_mtls else (LOCAL_TLS_CERT_SHA256 if enable_tls else None)
        ),
        "tls_subject_alt_names": LOCAL_TLS_SUBJECT_ALT_NAMES if enable_tls else [],
        "tls_minimum_version": LOCAL_TLS_MINIMUM_VERSION if enable_tls else None,
        "mtls_enabled": enable_mtls,
        "local_mtls_enabled": enable_mtls,
        "mtls_client_certificate_required": enable_mtls,
        "mtls_missing_client_cert_rejected": bool(
            enable_mtls and isinstance(mtls_reject, dict) and mtls_reject.get("transport_error") is True
        ),
        "mtls_verified_peer_count": mtls_summary.get("verified_peer_count"),
        "mtls_expected_peer_count": expected_endpoint_request_count if enable_mtls else 0,
        "mtls_peer_subjects": mtls_summary.get("peer_subjects"),
        "mtls_client_subject": LOCAL_MTLS_CLIENT_SUBJECT if enable_mtls else None,
        "mtls_all_peers_expected": mtls_summary.get("all_peers_expected"),
        "mtls_ca_certificate_sha256": LOCAL_MTLS_CA_CERT_SHA256 if enable_mtls else None,
        "mtls_client_certificate_sha256": LOCAL_MTLS_CLIENT_CERT_SHA256 if enable_mtls else None,
        "production_tls_enabled": False,
        "production_mtls_enabled": False,
        "production_auth_enabled": False,
        "target_model_parity": False,
        "g2_g3_gate_evidence": False,
        "correctness_passed": passed_count == len(checks),
    }
    result = {
        "version": 1,
        "record_kind": RECORD_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "endpoint": endpoint,
        "config": {key: value for key, value in config.items() if key != "auth_token"},
        "auth": {
            "mode": "local-bearer-token",
            "authorization_header_checked": True,
            "token_redacted": True,
            "production_auth": False,
        },
        "tls": {
            "enabled": enable_tls,
            "mode": "local-self-signed" if enable_tls and not enable_mtls else ("local-mutual-tls" if enable_mtls else "disabled"),
            "client_certificate_verified": enable_tls,
            "certificate_sha256": LOCAL_TLS_CERT_SHA256 if enable_tls and not enable_mtls else (LOCAL_MTLS_CA_CERT_SHA256 if enable_mtls else None),
            "subject_alt_names": LOCAL_TLS_SUBJECT_ALT_NAMES if enable_tls else [],
            "minimum_version": LOCAL_TLS_MINIMUM_VERSION if enable_tls else None,
            "private_key_redacted": True,
            "production_tls": False,
        },
        "mtls": {
            "enabled": enable_mtls,
            "mode": "local-mutual-tls" if enable_mtls else "disabled",
            "client_certificate_required": enable_mtls,
            "missing_client_certificate_rejected": bool(
                enable_mtls and isinstance(mtls_reject, dict) and mtls_reject.get("transport_error") is True
            ),
            "verified_peer_count": mtls_summary.get("verified_peer_count"),
            "expected_peer_count": expected_endpoint_request_count if enable_mtls else 0,
            "peer_subjects": mtls_summary.get("peer_subjects"),
            "client_subject": LOCAL_MTLS_CLIENT_SUBJECT if enable_mtls else None,
            "ca_certificate_sha256": LOCAL_MTLS_CA_CERT_SHA256 if enable_mtls else None,
            "client_certificate_sha256": LOCAL_MTLS_CLIENT_CERT_SHA256 if enable_mtls else None,
            "private_key_redacted": True,
            "production_mtls": False,
        },
        "lifecycle": lifecycle_summary,
        "backend": backend_summary,
        "target_fixture": target_fixture_summary,
        "serving_adapter": adapter,
        "responses": {
            "non_stream": non_stream,
            "stream": stream,
            "auth_reject": auth_reject,
            "mtls_reject": mtls_reject,
            "backpressure_holders": backpressure_holders,
            "backpressure_reject": backpressure_reject,
            "plan_reject": plan_reject,
            "bad_path": bad_path,
        },
        "checks": checks,
        "summary": summary,
        "ok": passed_count == len(checks),
        "note": (
            "Local HTTP/SSE serving smoke for the OpenAI-compatible endpoint path. "
            "This proves local endpoint request/response behavior, plan-integrity "
            "rejection, local bearer-token auth rejection, deterministic "
            "backpressure rejection, and local lifecycle cleanup. When target-fixture "
            "mode is enabled, it also proves deterministic local fixture loading and "
            "non-stream/stream parity only. TLS mode uses a local self-signed fixture "
            "certificate with client verification; mTLS mode additionally requires a local "
            "client certificate and records the peer identity. It is not product auth/mTLS, real frontier "
            "multi-host serving, or G2/G3 closure evidence."
        ),
    }
    write_json(output_path, result)
    return result


def validate_local_http_serving_smoke_fixture(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "local HTTP serving smoke is localhost-only and not target-model or multi-host gate evidence"
    ]
    if data.get("version") != 1:
        errors.append("version must be 1")
    if data.get("record_kind") != RECORD_KIND:
        errors.append(f"record_kind must be {RECORD_KIND}")
    if data.get("evidence_scope") != EVIDENCE_SCOPE:
        errors.append(f"evidence_scope must be {EVIDENCE_SCOPE}")
    adapter = data.get("serving_adapter")
    if not isinstance(adapter, dict):
        errors.append("serving_adapter must be an object")
    else:
        adapter_result = validate_serving_adapter_fixture(adapter)
        errors.extend(f"serving_adapter: {error}" for error in adapter_result["errors"])
        warnings.extend(f"serving_adapter: {warning}" for warning in adapter_result["warnings"])
    backend = data.get("backend")
    if not isinstance(backend, dict):
        errors.append("backend must be an object")
        backend = {}
    if backend.get("backend") != "FornaxBackend":
        errors.append("backend.backend must be FornaxBackend")
    backend_mode = backend.get("mode")
    if backend_mode not in {"local-http-smoke", "local-http-target-fixture-smoke"}:
        errors.append("backend.mode must be local-http-smoke or local-http-target-fixture-smoke")
    if backend.get("engine_trait_compatible") is not True:
        errors.append("backend.engine_trait_compatible must be true")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}
    max_inflight = summary.get("max_inflight")
    expected_backend_request_count = 2 + max_inflight if isinstance(max_inflight, int) and not isinstance(max_inflight, bool) and max_inflight > 0 else None
    if expected_backend_request_count is None:
        errors.append("summary.max_inflight must be a positive integer")
    elif backend.get("request_count") != expected_backend_request_count:
        errors.append(f"backend.request_count must be {expected_backend_request_count}")
    target_fixture = data.get("target_fixture")
    target_fixture_enabled = backend_mode == "local-http-target-fixture-smoke"
    if target_fixture_enabled:
        if backend.get("target_model_loaded") is not True:
            errors.append("backend.target_model_loaded must be true in target-fixture mode")
        if backend.get("target_model_scope") != "local-fixture-only":
            errors.append("backend.target_model_scope must be local-fixture-only in target-fixture mode")
        if backend.get("target_model_parity") is not True:
            errors.append("backend.target_model_parity must be true in target-fixture mode")
        if backend.get("real_frontier_model_loaded") is not False:
            errors.append("backend.real_frontier_model_loaded must be false")
        if backend.get("real_frontier_model_parity") is not False:
            errors.append("backend.real_frontier_model_parity must be false")
        if not isinstance(target_fixture, dict):
            errors.append("target_fixture must be an object in target-fixture mode")
            target_fixture = {}
        if target_fixture.get("scope") != "local-fixture-only":
            errors.append("target_fixture.scope must be local-fixture-only")
        if target_fixture.get("fixture_model_id") != TARGET_FIXTURE_MODEL_ID:
            errors.append(f"target_fixture.fixture_model_id must be {TARGET_FIXTURE_MODEL_ID}")
        if target_fixture.get("loaded") is not True:
            errors.append("target_fixture.loaded must be true")
        if target_fixture.get("parity") is not True:
            errors.append("target_fixture.parity must be true")
        if target_fixture.get("non_stream_matches_stream") is not True:
            errors.append("target_fixture.non_stream_matches_stream must be true")
        if target_fixture.get("stream_run_count", 0) < 1:
            errors.append("target_fixture.stream_run_count must be at least 1")
        if target_fixture.get("non_stream_run_count", 0) < 1:
            errors.append("target_fixture.non_stream_run_count must be at least 1")
        if not _is_sha256(target_fixture.get("template_hash")):
            errors.append("target_fixture.template_hash must be a sha256 hash")
        if not _is_sha256(target_fixture.get("tokenizer_hash")):
            errors.append("target_fixture.tokenizer_hash must be a sha256 hash")
        if target_fixture.get("real_frontier_model_loaded") is not False:
            errors.append("target_fixture.real_frontier_model_loaded must be false")
        if target_fixture.get("real_frontier_model_parity") is not False:
            errors.append("target_fixture.real_frontier_model_parity must be false")
    else:
        if backend.get("target_model_loaded") is not False:
            errors.append("backend.target_model_loaded must be false")
        if backend.get("target_model_parity") is not False:
            errors.append("backend.target_model_parity must be false")
        if target_fixture is not None:
            errors.append("target_fixture must be null unless target-fixture mode is enabled")
    config = data.get("config")
    if isinstance(config, dict) and "auth_token" in config:
        errors.append("config.auth_token must be redacted")
    lifecycle = data.get("lifecycle")
    if not isinstance(lifecycle, dict):
        errors.append("lifecycle must be an object")
        lifecycle = {}
    if lifecycle.get("mode") != "local-http-lifecycle-smoke":
        errors.append("lifecycle.mode must be local-http-lifecycle-smoke")
    if lifecycle.get("resource_kinds") != list(LOCAL_LIFECYCLE_RESOURCE_KINDS):
        errors.append("lifecycle.resource_kinds must match local lifecycle resource kinds")
    lifecycle_events = lifecycle.get("events")
    if not isinstance(lifecycle_events, list) or not lifecycle_events:
        errors.append("lifecycle.events must be a non-empty list")
        lifecycle_events = []
    if lifecycle.get("all_required_resources_released") is not True:
        errors.append("lifecycle.all_required_resources_released must be true")
    if lifecycle.get("single_owner_preserved") is not True:
        errors.append("lifecycle.single_owner_preserved must be true")
    auth = data.get("auth")
    if not isinstance(auth, dict):
        errors.append("auth must be an object")
        auth = {}
    if auth.get("mode") != "local-bearer-token":
        errors.append("auth.mode must be local-bearer-token")
    if auth.get("authorization_header_checked") is not True:
        errors.append("auth.authorization_header_checked must be true")
    if auth.get("token_redacted") is not True:
        errors.append("auth.token_redacted must be true")
    if auth.get("production_auth") is not False:
        errors.append("auth.production_auth must be false")
    tls = data.get("tls")
    if not isinstance(tls, dict):
        errors.append("tls must be an object")
        tls = {}
    tls_enabled = summary.get("tls_enabled") is True
    mtls_enabled = summary.get("mtls_enabled") is True
    if tls_enabled:
        expected_tls_mode = "local-mutual-tls" if mtls_enabled else "local-self-signed"
        if summary.get("local_tls_enabled") is not True:
            errors.append("summary.local_tls_enabled must be true when TLS is enabled")
        if summary.get("tls_client_verified") is not True:
            errors.append("summary.tls_client_verified must be true when TLS is enabled")
        if summary.get("tls_mode") != expected_tls_mode:
            errors.append(f"summary.tls_mode must be {expected_tls_mode} when TLS is enabled")
        if not _is_sha256(summary.get("tls_certificate_sha256")):
            errors.append("summary.tls_certificate_sha256 must be a sha256 hash when TLS is enabled")
        if summary.get("tls_subject_alt_names") != LOCAL_TLS_SUBJECT_ALT_NAMES:
            errors.append("summary.tls_subject_alt_names must match local TLS SANs")
        if summary.get("tls_minimum_version") != LOCAL_TLS_MINIMUM_VERSION:
            errors.append("summary.tls_minimum_version must match local TLS minimum")
        if tls.get("enabled") is not True:
            errors.append("tls.enabled must be true when summary.tls_enabled is true")
        if tls.get("mode") != expected_tls_mode:
            errors.append(f"tls.mode must be {expected_tls_mode} when TLS is enabled")
        if tls.get("client_certificate_verified") is not True:
            errors.append("tls.client_certificate_verified must be true when TLS is enabled")
        if tls.get("certificate_sha256") != summary.get("tls_certificate_sha256"):
            errors.append("tls.certificate_sha256 must match summary.tls_certificate_sha256")
        if tls.get("subject_alt_names") != LOCAL_TLS_SUBJECT_ALT_NAMES:
            errors.append("tls.subject_alt_names must match local TLS SANs")
        if tls.get("private_key_redacted") is not True:
            errors.append("tls.private_key_redacted must be true")
    else:
        if summary.get("local_tls_enabled") is not False:
            errors.append("summary.local_tls_enabled must be false when TLS is disabled")
        if summary.get("tls_client_verified") is not False:
            errors.append("summary.tls_client_verified must be false when TLS is disabled")
        if summary.get("tls_mode") != "disabled":
            errors.append("summary.tls_mode must be disabled when TLS is disabled")
        if tls.get("enabled") is not False:
            errors.append("tls.enabled must be false when summary.tls_enabled is false")
        if tls.get("mode") != "disabled":
            errors.append("tls.mode must be disabled when TLS is disabled")
    if tls.get("production_tls") is not False:
        errors.append("tls.production_tls must be false")
    mtls = data.get("mtls")
    if not isinstance(mtls, dict):
        errors.append("mtls must be an object")
        mtls = {}
    if mtls_enabled:
        expected_peer_count = summary.get("mtls_expected_peer_count")
        if summary.get("local_mtls_enabled") is not True:
            errors.append("summary.local_mtls_enabled must be true when mTLS is enabled")
        if summary.get("mtls_client_certificate_required") is not True:
            errors.append("summary.mtls_client_certificate_required must be true")
        if summary.get("mtls_missing_client_cert_rejected") is not True:
            errors.append("summary.mtls_missing_client_cert_rejected must be true")
        if summary.get("mtls_client_subject") != LOCAL_MTLS_CLIENT_SUBJECT:
            errors.append("summary.mtls_client_subject must match local mTLS client subject")
        if not isinstance(expected_peer_count, int) or expected_peer_count <= 0:
            errors.append("summary.mtls_expected_peer_count must be a positive integer")
        elif summary.get("mtls_verified_peer_count") != expected_peer_count:
            errors.append("summary.mtls_verified_peer_count must match summary.mtls_expected_peer_count")
        peer_subjects = summary.get("mtls_peer_subjects")
        if not isinstance(peer_subjects, list) or not peer_subjects:
            errors.append("summary.mtls_peer_subjects must be a non-empty list")
        elif any(subject != LOCAL_MTLS_CLIENT_SUBJECT for subject in peer_subjects):
            errors.append("summary.mtls_peer_subjects must all match local mTLS client subject")
        if summary.get("mtls_all_peers_expected") is not True:
            errors.append("summary.mtls_all_peers_expected must be true")
        if not _is_sha256(summary.get("mtls_ca_certificate_sha256")):
            errors.append("summary.mtls_ca_certificate_sha256 must be a sha256 hash")
        if not _is_sha256(summary.get("mtls_client_certificate_sha256")):
            errors.append("summary.mtls_client_certificate_sha256 must be a sha256 hash")
        if mtls.get("enabled") is not True:
            errors.append("mtls.enabled must be true when summary.mtls_enabled is true")
        if mtls.get("mode") != "local-mutual-tls":
            errors.append("mtls.mode must be local-mutual-tls when mTLS is enabled")
        if mtls.get("client_certificate_required") is not True:
            errors.append("mtls.client_certificate_required must be true")
        if mtls.get("missing_client_certificate_rejected") is not True:
            errors.append("mtls.missing_client_certificate_rejected must be true")
        if expected_peer_count is not None and mtls.get("verified_peer_count") != expected_peer_count:
            errors.append("mtls.verified_peer_count must match summary.mtls_expected_peer_count")
        if mtls.get("client_subject") != LOCAL_MTLS_CLIENT_SUBJECT:
            errors.append("mtls.client_subject must match local mTLS client subject")
        if mtls.get("private_key_redacted") is not True:
            errors.append("mtls.private_key_redacted must be true")
    else:
        if summary.get("local_mtls_enabled") is not False:
            errors.append("summary.local_mtls_enabled must be false when mTLS is disabled")
        if mtls.get("enabled") is not False:
            errors.append("mtls.enabled must be false when summary.mtls_enabled is false")
        if mtls.get("mode") != "disabled":
            errors.append("mtls.mode must be disabled when mTLS is disabled")
    if mtls.get("production_mtls") is not False:
        errors.append("mtls.production_mtls must be false")
    responses = data.get("responses")
    if not isinstance(responses, dict):
        errors.append("responses must be an object")
        responses = {}
    auth_reject = responses.get("auth_reject") if isinstance(responses, dict) else None
    if not isinstance(auth_reject, dict):
        errors.append("responses.auth_reject must be an object")
    else:
        if auth_reject.get("status") != 401:
            errors.append("responses.auth_reject.status must be 401")
        if auth_reject.get("body", {}).get("error", {}).get("code") != "endpoint_auth_required":
            errors.append("responses.auth_reject error code must be endpoint_auth_required")
    mtls_reject = responses.get("mtls_reject") if isinstance(responses, dict) else None
    if mtls_enabled:
        if not isinstance(mtls_reject, dict):
            errors.append("responses.mtls_reject must be an object when mTLS is enabled")
        elif mtls_reject.get("transport_error") is not True:
            errors.append("responses.mtls_reject.transport_error must be true when mTLS is enabled")
    elif mtls_reject is not None:
        errors.append("responses.mtls_reject must be null unless mTLS is enabled")
    backpressure_holders = responses.get("backpressure_holders") if isinstance(responses, dict) else None
    if not isinstance(backpressure_holders, list):
        errors.append("responses.backpressure_holders must be a list")
        backpressure_holders = []
    elif expected_backend_request_count is not None and len(backpressure_holders) != max_inflight:
        errors.append("responses.backpressure_holders length must match summary.max_inflight")
    for index, holder in enumerate(backpressure_holders):
        if not isinstance(holder, dict):
            errors.append(f"responses.backpressure_holders[{index}] must be an object")
        elif holder.get("status") != 200:
            errors.append(f"responses.backpressure_holders[{index}].status must be 200")
    backpressure_reject = responses.get("backpressure_reject") if isinstance(responses, dict) else None
    if not isinstance(backpressure_reject, dict):
        errors.append("responses.backpressure_reject must be an object")
    else:
        if backpressure_reject.get("status") != 429:
            errors.append("responses.backpressure_reject.status must be 429")
        if backpressure_reject.get("body", {}).get("error", {}).get("code") != "backpressure_queue_full":
            errors.append("responses.backpressure_reject error code must be backpressure_queue_full")
    checks = data.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("checks must be a non-empty list")
        checks = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"checks[{index}] must be an object")
            continue
        if not check.get("name"):
            errors.append(f"checks[{index}].name must be set")
        if check.get("ok") is not True:
            errors.append(f"checks[{index}] {check.get('name', '<unknown>')} must pass")
    check_names = {check.get("name") for check in checks if isinstance(check, dict)}
    if tls_enabled and "local-tls-handshake" not in check_names:
        errors.append("checks must include local-tls-handshake when TLS is enabled")
    if mtls_enabled and "local-mtls-node-identity" not in check_names:
        errors.append("checks must include local-mtls-node-identity when mTLS is enabled")
    passed_count = sum(1 for check in checks if isinstance(check, dict) and check.get("ok") is True)
    if summary.get("check_count") != len(checks):
        errors.append("summary.check_count must match checks")
    if summary.get("passed_count") != passed_count:
        errors.append("summary.passed_count must match checks")
    if summary.get("http_endpoint_started") is not True:
        errors.append("summary.http_endpoint_started must be true")
    if summary.get("non_stream_status") != 200:
        errors.append("summary.non_stream_status must be 200")
    if summary.get("stream_status") != 200:
        errors.append("summary.stream_status must be 200")
    if summary.get("sse_done_seen") is not True:
        errors.append("summary.sse_done_seen must be true")
    if summary.get("auth_reject_status") != 401:
        errors.append("summary.auth_reject_status must be 401")
    if summary.get("endpoint_auth_rejected") is not True:
        errors.append("summary.endpoint_auth_rejected must be true")
    if summary.get("backpressure_status") != 429:
        errors.append("summary.backpressure_status must be 429")
    if summary.get("backpressure_rejected") is not True:
        errors.append("summary.backpressure_rejected must be true")
    if expected_backend_request_count is not None and summary.get("backpressure_holder_count") != max_inflight:
        errors.append("summary.backpressure_holder_count must match summary.max_inflight")
    holder_statuses = summary.get("backpressure_holder_statuses")
    if expected_backend_request_count is not None:
        if not isinstance(holder_statuses, list) or holder_statuses != [200] * max_inflight:
            errors.append("summary.backpressure_holder_statuses must all be 200")
    if summary.get("backpressure_holders_completed") is not True:
        errors.append("summary.backpressure_holders_completed must be true")
    if expected_backend_request_count is not None and summary.get("max_observed_inflight") != max_inflight:
        errors.append("summary.max_observed_inflight must match summary.max_inflight")
    if summary.get("backpressure_reject_count") != 1:
        errors.append("summary.backpressure_reject_count must be 1")
    if expected_backend_request_count is not None and summary.get("inflight_cleanup_count") != expected_backend_request_count:
        errors.append(f"summary.inflight_cleanup_count must be {expected_backend_request_count}")
    if summary.get("failure_semantics_verified") is not True:
        errors.append("summary.failure_semantics_verified must be true")
    if expected_backend_request_count is not None:
        expected_resource_count = expected_backend_request_count * len(LOCAL_LIFECYCLE_RESOURCE_KINDS)
        if summary.get("lifecycle_tracked") is not True:
            errors.append("summary.lifecycle_tracked must be true")
        if summary.get("lifecycle_request_count") != expected_backend_request_count:
            errors.append(f"summary.lifecycle_request_count must be {expected_backend_request_count}")
        if summary.get("lifecycle_rejected_request_count") != 4:
            errors.append("summary.lifecycle_rejected_request_count must be 4")
        if summary.get("lifecycle_cleanup_count") != expected_backend_request_count:
            errors.append(f"summary.lifecycle_cleanup_count must be {expected_backend_request_count}")
        if summary.get("lifecycle_resource_allocated_count") != expected_resource_count:
            errors.append(f"summary.lifecycle_resource_allocated_count must be {expected_resource_count}")
        if summary.get("lifecycle_resource_released_count") != expected_resource_count:
            errors.append(f"summary.lifecycle_resource_released_count must be {expected_resource_count}")
        if summary.get("lifecycle_active_resource_count") != 0:
            errors.append("summary.lifecycle_active_resource_count must be 0")
        if summary.get("lifecycle_all_released") is not True:
            errors.append("summary.lifecycle_all_released must be true")
        if summary.get("lifecycle_single_owner_preserved") is not True:
            errors.append("summary.lifecycle_single_owner_preserved must be true")
        if lifecycle.get("accepted_request_count") != expected_backend_request_count:
            errors.append(f"lifecycle.accepted_request_count must be {expected_backend_request_count}")
        if lifecycle.get("rejected_request_count") != 4:
            errors.append("lifecycle.rejected_request_count must be 4")
        if lifecycle.get("cleanup_count") != expected_backend_request_count:
            errors.append(f"lifecycle.cleanup_count must be {expected_backend_request_count}")
        if lifecycle.get("resource_allocated_count") != expected_resource_count:
            errors.append(f"lifecycle.resource_allocated_count must be {expected_resource_count}")
        if lifecycle.get("resource_released_count") != expected_resource_count:
            errors.append(f"lifecycle.resource_released_count must be {expected_resource_count}")
        if lifecycle.get("active_resource_count") != 0:
            errors.append("lifecycle.active_resource_count must be 0")
    if summary.get("plan_integrity_rejected") is not True:
        errors.append("summary.plan_integrity_rejected must be true")
    if summary.get("bad_path_rejected") is not True:
        errors.append("summary.bad_path_rejected must be true")
    if summary.get("fornax_backend_integrated") is not True:
        errors.append("summary.fornax_backend_integrated must be true")
    if expected_backend_request_count is not None and summary.get("backend_request_count") != expected_backend_request_count:
        errors.append(f"summary.backend_request_count must be {expected_backend_request_count}")
    if target_fixture_enabled and expected_backend_request_count is not None:
        if summary.get("backend_target_model_loaded") is not True:
            errors.append("summary.backend_target_model_loaded must be true in target-fixture mode")
        if summary.get("backend_target_model_scope") != "local-fixture-only":
            errors.append("summary.backend_target_model_scope must be local-fixture-only in target-fixture mode")
        if summary.get("backend_target_model_parity") is not True:
            errors.append("summary.backend_target_model_parity must be true in target-fixture mode")
        if summary.get("target_fixture_loaded") is not True:
            errors.append("summary.target_fixture_loaded must be true in target-fixture mode")
        if summary.get("target_fixture_parity") is not True:
            errors.append("summary.target_fixture_parity must be true in target-fixture mode")
        if summary.get("target_fixture_run_count") != expected_backend_request_count:
            errors.append(f"summary.target_fixture_run_count must be {expected_backend_request_count}")
        if summary.get("target_fixture_non_stream_matches_stream") is not True:
            errors.append("summary.target_fixture_non_stream_matches_stream must be true")
        if not _is_sha256(summary.get("target_fixture_template_hash")):
            errors.append("summary.target_fixture_template_hash must be a sha256 hash")
        if not _is_sha256(summary.get("target_fixture_tokenizer_hash")):
            errors.append("summary.target_fixture_tokenizer_hash must be a sha256 hash")
        check_names = {check.get("name") for check in checks if isinstance(check, dict)}
        if "target-fixture-parity" not in check_names:
            errors.append("checks must include target-fixture-parity in target-fixture mode")
    if not target_fixture_enabled:
        if summary.get("backend_target_model_loaded") is not False:
            errors.append("summary.backend_target_model_loaded must be false")
        if summary.get("target_fixture_loaded") is not False:
            errors.append("summary.target_fixture_loaded must be false")
        if summary.get("target_fixture_parity") is not False:
            errors.append("summary.target_fixture_parity must be false")
    if summary.get("real_frontier_model_loaded") is not False:
        errors.append("summary.real_frontier_model_loaded must be false")
    if summary.get("real_frontier_model_parity") is not False:
        errors.append("summary.real_frontier_model_parity must be false")
    if summary.get("engine_trait_compatible") is not True:
        errors.append("summary.engine_trait_compatible must be true")
    if summary.get("engine_result_emitted") is not True:
        errors.append("summary.engine_result_emitted must be true")
    if summary.get("live_http_endpoint") is not True:
        errors.append("summary.live_http_endpoint must be true")
    if summary.get("localhost_only") is not True:
        errors.append("summary.localhost_only must be true")
    if summary.get("local_auth_enabled") is not True:
        errors.append("summary.local_auth_enabled must be true")
    if summary.get("auth_token_redacted") is not True:
        errors.append("summary.auth_token_redacted must be true")
    if summary.get("production_tls_enabled") is not False:
        errors.append("summary.production_tls_enabled must be false for local smoke")
    if summary.get("production_mtls_enabled") is not False:
        errors.append("summary.production_mtls_enabled must be false for local smoke")
    if summary.get("production_auth_enabled") is not False:
        errors.append("summary.production_auth_enabled must be false for local smoke")
    if summary.get("target_model_parity") is not False:
        errors.append("summary.target_model_parity must be false")
    if summary.get("g2_g3_gate_evidence") is not False:
        errors.append("summary.g2_g3_gate_evidence must be false")
    if summary.get("correctness_passed") is not True:
        errors.append("summary.correctness_passed must be true")
    if data.get("ok") is not True:
        errors.append("ok must be true")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "check_count": summary.get("check_count"),
            "passed_count": passed_count,
            "endpoint": summary.get("endpoint"),
            "sse_chunk_count": summary.get("sse_chunk_count"),
            "endpoint_auth_rejected": summary.get("endpoint_auth_rejected") is True,
            "tls_enabled": summary.get("tls_enabled") is True,
            "local_tls_enabled": summary.get("local_tls_enabled") is True,
            "tls_client_verified": summary.get("tls_client_verified") is True,
            "production_tls_enabled": summary.get("production_tls_enabled") is False,
            "mtls_enabled": summary.get("mtls_enabled") is True,
            "local_mtls_enabled": summary.get("local_mtls_enabled") is True,
            "mtls_missing_client_cert_rejected": summary.get("mtls_missing_client_cert_rejected") is True,
            "production_mtls_enabled": summary.get("production_mtls_enabled") is False,
            "backpressure_rejected": summary.get("backpressure_rejected") is True,
            "backpressure_reject_count": summary.get("backpressure_reject_count"),
            "failure_semantics_verified": summary.get("failure_semantics_verified") is True,
            "lifecycle_tracked": summary.get("lifecycle_tracked") is True,
            "lifecycle_request_count": summary.get("lifecycle_request_count"),
            "lifecycle_all_released": summary.get("lifecycle_all_released") is True,
            "plan_integrity_rejected": summary.get("plan_integrity_rejected") is True,
            "fornax_backend_integrated": summary.get("fornax_backend_integrated") is True,
            "backend_request_count": summary.get("backend_request_count"),
            "engine_trait_compatible": summary.get("engine_trait_compatible") is True,
            "engine_result_emitted": summary.get("engine_result_emitted") is True,
            "backend_target_model_loaded": summary.get("backend_target_model_loaded") is True,
            "target_fixture_parity": summary.get("target_fixture_parity") is True,
            "real_frontier_model_parity": summary.get("real_frontier_model_parity") is True,
            "live_http_endpoint": summary.get("live_http_endpoint") is True,
            "target_model_parity": summary.get("target_model_parity") is True,
            "g2_g3_gate_evidence": summary.get("g2_g3_gate_evidence") is True,
        },
    }


def validate_local_http_serving_smoke(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    try:
        data = read_json(fixture_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"invalid local HTTP serving smoke artifact: {exc}"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    if not isinstance(data, dict):
        return {
            "ok": False,
            "errors": ["local HTTP serving smoke artifact must be a JSON object"],
            "warnings": [],
            "summary": {},
            "fixture": str(fixture_path),
        }
    result = validate_local_http_serving_smoke_fixture(data)
    result["fixture"] = str(fixture_path)
    return result
