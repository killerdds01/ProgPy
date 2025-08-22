// static/js/profile.js
// Utility semplice: alterna visibilità dei campi password quando l'utente
// clicca un bottone con classe .eye-btn e attributo data-target che indica
// il "name" dell'input da controllare.
// Esempio markup:
//   <input type="password" name="password" ...>
//   <button type="button" class="eye-btn" data-target="password">👁</button>

document.addEventListener('DOMContentLoaded', () => {
  // Seleziono tutti i bottoni con classe .eye-btn (può essercene più di uno)
  document.querySelectorAll('.eye-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      // "data-target" contiene il name dell'input da alternare
      const targetName = btn.getAttribute('data-target');
      if (!targetName) return;

      // Cerco l'input per name=...
      const input = document.querySelector(`input[name="${targetName}"]`);
      if (!input) return;

      // Alterno il type: password <-> text
      input.type = (input.type === 'password') ? 'text' : 'password';
    });
  });
});
