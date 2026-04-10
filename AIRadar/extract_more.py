import ast
import sys

def extract_node(filename, names):
    with open(filename, "r") as f:
        source = f.read()
    
    module = ast.parse(source)
    extracted = []
    
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if node.name in names:
                extracted.append(ast.unparse(node))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    extracted.append(ast.unparse(node))
    return "\n\n".join(extracted)

def main():
    names_g2c = [
        "FiLMConvBlock",
        "SEBlock",
        "ComplexConvBlock",
        "ComplexToReal",
        "CommAdapter",
        "CommHeadLLR"
    ]
    
    code_g2c = extract_node("/Developer/AIsensing/AIRadar/AIradar_comm_model_g2c.py", names_g2c)
    
    with open("/Developer/AIsensing/AIRadar/extracted_more.py", "w") as f:
        f.write(code_g2c)

if __name__ == "__main__":
    main()
