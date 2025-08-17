// Toggle occhio mostra/nascondi password
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.eye-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetName = btn.getAttribute('data-target');
      if (!targetName) return;
      const input = document.querySelector(`input[name="${targetName}"]`);
      if (!input) return;
      input.type = (input.type === 'password') ? 'text' : 'password';
    });
  });
});
