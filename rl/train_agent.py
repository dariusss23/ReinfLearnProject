"""Antrenare: legatura dintre agent si CybORG.

Aici intra:
  - crearea mediului (BlueEnterpriseWrapper, 5 agenti blue)
  - padding observatii la 210 si masca de actiuni (82 vs 242)
  - bucla de colectare a experientei si apelul agent.update(...)
  - tratarea actiunilor cu durata variabila, scalarea recompensei
  - salvarea checkpoint-urilor in checkpoints/
"""
