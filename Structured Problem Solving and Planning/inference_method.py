from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE

def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """
    
    ### START: Your code
    '''
    PSEUDO CODE FROM TEXTBOOK
    
    function PL-FC-ENTAILS?(KB, q) returns true or false
    inputs: KB, the knowledge base, a set of propositional definite clauses 
            q, the query, a proposition symbol
    
    count ←a table, where count [c] is the number of symbols in c’s premise
    inferred ←a table,where inferred[s] is initially false for all symbols
    agenda ←a queue of symbols, initially symbols known to be true in KB
    
    while agenda is not empty do
        p←POP(agenda)
        if p = q then return true
        if inferred[p] = false then
            inferred[p]←true
            for each clause c in KB where p is in c.PREMISE do
                decrement count [c]
                if count [c] = 0 then add c.CONCLUSION to agenda
    return false
    '''
    
    agenda = deque()
    for elem in known_symbols:
        agenda.append(elem)
    
    inferred = dict()
    for elem in symbols_list:
        inferred[elem] = False
    
    count = dict()
    for elem in symbols_list:
        count[elem] = 0 
    
    while len(agenda)> 0:
        p = agenda.popleft()
        if p == query:
            return True
        
        if inferred[p] == False:
            inferred[p] = True 
        
        for c in KB_clauses:
            if p in c.body:
                count[c.conclusion] -= 1 
            if count[c.conclusion] == 0:
                agenda.append(c.conclusion)
    
    return False # remove line if needed
    ### END: Your code


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1,2,9,4,5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9,4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
