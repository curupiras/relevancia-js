import re

def remove_html(html):
    return re.sub("<[^>]*>", "", html).strip()