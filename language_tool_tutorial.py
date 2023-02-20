import language_tool_python
tool = language_tool_python.LanguageToolPublicAPI('en-US') # starts a process
# do stuff with `tool`
text = 'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'
matches = tool.check(text)
len(matches)
print(matches[0].ruleId, matches[0].replacements)
print(matches[1].ruleId, matches[1].replacements)

tool = language_tool_python.LanguageToolPublicAPI('ja-JA') # starts a process
# do stuff with `tool`
text = '私はラーメンべるのが好きではありません'
matches = tool.check(text)
print(len(matches))
for i in range(len(matches)):
    print(matches[i].ruleId, matches[0].replacements)

tool.close() # explicitly shut off the LanguageTool