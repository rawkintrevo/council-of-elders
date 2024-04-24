from google.ai.generativelanguage_v1beta.types.content import Content as gContent
def update_role(items, self_name):
    for item in items:
        if isinstance(item, gContent):
            item = {'role': item.role, 'parts': item.parts}
        if item['role'] == self_name:
            item['role'] = 'assistant'
        else:
            item['role'] = 'user'
    return items

def merge_items_by_role(items):
    if not items:
        return []

    # Initialize the output list with the first item
    output = [items[0]]

    for item in items[1:]:  # Start from the second item
        if isinstance(item, gContent):
            item = {'role': item.role, 'parts': item.parts}
        last_item = output[-1]
        if isinstance(last_item, gContent):
            last_item = {'role': last_item.role, 'parts': last_item.parts}
        if item['role'] == last_item['role']:
            # If the current item's role matches the last item in the output,
            # append the current content to the last item's content with a newline
            if 'content' in last_item:
                last_item['content'] += f"\n{item['content']}"
            if 'parts' in last_item:
                last_item['parts'].extend(item['parts'])
        else:
            # If the role does not match, simply add the item to the output list
            output.append(item)

    return output
