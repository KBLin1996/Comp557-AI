Note:
If you are using Python version 3.8 or 3.9, you might see this error:
module 'cgi' has no attribute 'escape'
This error is caused line 294 in grading.py "message = cgi.escape(message)"
The module cgi is updated in higher Python version, so
Please downgrade your Python version to 3.7 (or lower) OR 
replace "cgi.escape" with "html.escape" on line 294 (please update the import accordingly) to solve it.