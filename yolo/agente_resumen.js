'use strict';

/**
 * Agente de resumen para Groq.
 *
 * Recibe por stdin un JSON con las métricas calculadas por dashboard.py,
 * incluyendo la distribución numérica extraída del mapa de calor.
 * No recibe imágenes ni datos biométricos.
 */

function leerStdin() {
  return new Promise((resolve, reject) => {
    let contenido = '';

    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (fragmento) => {
      contenido += fragmento;
    });
    process.stdin.on('end', () => resolve(contenido));
    process.stdin.on('error', reject);
  });
}

function validarMetricas(datos) {
  const campos = [
    'fecha',
    'horaInicio',
    'horaFin',
    'entradas',
    'salidas',
    'balanceNeto',
    'horaPico',
    'entradasHoraPico',
    'mapaCalorDisponible',
    'zonaMayorConcentracion',
    'concentracionZonaMayor',
    'distribucionZonas',
    'muestrasMapaCalor',
    'promedioPersonasDetectadas'
  ];

  for (const campo of campos) {
    if (!(campo in datos)) {
      throw new Error(`Falta el campo obligatorio: ${campo}`);
    }
  }

  if (
    datos.distribucionZonas === null ||
    typeof datos.distribucionZonas !== 'object' ||
    Array.isArray(datos.distribucionZonas)
  ) {
    throw new Error('distribucionZonas debe ser un objeto.');
  }
}

async function generarResumen(datos) {
  const apiKey = process.env.GROQ_API_KEY;
  const modelo = process.env.GROQ_MODEL;

  if (!apiKey) {
    throw new Error('No se configuró GROQ_API_KEY.');
  }

  if (!modelo) {
    throw new Error('No se configuró GROQ_MODEL.');
  }

  const promptSistema = [
    'Eres un agente analista de flujo y distribución espacial de clientes para un establecimiento comercial.',
    'Genera un resumen ejecutivo claro, breve y basado únicamente en los datos entregados.',
    'No inventes causas, eventos, porcentajes ni información externa.',
    'La hora pico representa la hora con mayor cantidad de entradas registradas, no necesariamente el máximo aforo simultáneo.',
    'Los valores del mapa de calor son porcentajes relativos de actividad visual por zona; no representan temperatura corporal ni datos obtenidos de sensores térmicos.',
    'Cuando mapaCalorDisponible sea verdadero, menciona la zona con mayor concentración relativa y formula una recomendación operativa prudente sobre la distribución del espacio.',
    'Cuando mapaCalorDisponible sea falso, indica que no hubo suficientes muestras para analizar la distribución espacial.',
    'Incluye: panorama general, hora con más entradas, interpretación del balance neto, análisis espacial y una recomendación.',
    'Cuando el volumen de datos sea bajo, aclara que las conclusiones son preliminares.',
    'Escribe entre 110 y 170 palabras en español y no uses tablas ni formato JSON.'
  ].join(' ');

  const promptUsuario = `Analiza estas métricas del flujo de clientes:\n${JSON.stringify(datos, null, 2)}`;

  const respuesta = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: modelo,
      temperature: 0.2,
      max_completion_tokens: 450,
      messages: [
        { role: 'system', content: promptSistema },
        { role: 'user', content: promptUsuario }
      ]
    })
  });

  const cuerpo = await respuesta.json().catch(() => ({}));

  if (!respuesta.ok) {
    const detalle = cuerpo?.error?.message || `Groq respondió con HTTP ${respuesta.status}`;
    throw new Error(detalle);
  }

  const resumen = cuerpo?.choices?.[0]?.message?.content?.trim();

  if (!resumen) {
    throw new Error('Groq no devolvió contenido para el resumen.');
  }

  return resumen;
}

async function main() {
  try {
    const entrada = await leerStdin();
    const datos = JSON.parse(entrada);
    validarMetricas(datos);

    const resumen = await generarResumen(datos);
    process.stdout.write(JSON.stringify({ ok: true, resumen }));
  } catch (error) {
    process.stdout.write(JSON.stringify({
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    }));
    process.exitCode = 1;
  }
}

main();
