# Frenchrs

Biblioteca Rust de alto desempenho para precificação de ativos e análise financeira, construída sobre a infraestrutura econométrica robusta do [Greeners](https://crates.io/crates/greeners).

## 📊 Modelos Implementados

### Modelos Clássicos
- **CAPM** (Capital Asset Pricing Model, 1964)
  - Modelo fundamental de precificação baseado no risco sistemático
  - Retorno esperado = Rf + β(Rm - Rf)

### Modelos Fama-French
- **Fama-French 3 Factor** (1993)
  - CAPM + fatores tamanho (SMB) e valor (HML)
  - Melhora significativa no poder explicativo

- **Fama-French 5 Factor** (2015)
  - FF3 + fatores rentabilidade (RMW) e investimento (CMA)
  - Estado da arte em precificação de ativos

- **Fama-French 6 Factor** (2023)
  - FF5 + fator momentum (UMD/Up Minus Down)
  - Modelo mais completo disponível

### Modelos Multi-fatoriais
- **Carhart 4 Factor** (1997)
  - FF3 + fator momentum (MOM)
  - Popular para análise de fundos de investimento

- **APT** (Arbitrage Pricing Theory, 1976)
  - Framework genérico com N fatores arbitrários
  - Máxima flexibilidade para pesquisa e modelos customizados

### Métricas de Risco
- **IVOL (Idiosyncratic Volatility)**
  - Volatilidade específica não explicada pelos fatores
  - IVOL anualizado (diário e mensal)
  - Estatísticas completas dos resíduos (skewness, kurtosis)
  - Teste de normalidade Jarque-Bera

- **Tracking Error Analysis**
  - Tracking error ex-post
  - Information ratio
  - Rolling tracking error (janela de 12 períodos)
  - Métricas de qualidade do ajuste (RMSE, MAE, correlação)

### Análise Temporal
- **Rolling Betas**
  - Análise de janelas móveis para CAPM e Fama-French 3
  - Evolução temporal de alphas e betas
  - Estatísticas de estabilidade (CV, tendência, autocorrelação)
  - Identificação de mudanças estruturais
  - Classificação automática de estabilidade

## 🚀 Características

- ✅ **Alto Desempenho**: Construído em Rust com BLAS/LAPACK
- ✅ **Estatisticamente Robusto**: Múltiplos tipos de erros padrão (HC0-HC4, Newey-West, Clustering)
- ✅ **Completo**: Estatísticas t, p-values, intervalos de confiança, métricas de performance
- ✅ **Flexível**: Suporte para DataFrame e arrays ndarray
- ✅ **Bem Testado**: >87 testes unitários e de integração
- ✅ **Bem Documentado**: Exemplos completos e documentação inline

## 📦 Instalação

```toml
[dependencies]
frenchrs = "0.1.0"
greeners = "1.3.2"
ndarray = "0.17.1"
```

## 📚 Uso Básico

### CAPM

```rust
use frenchrs::CAPM;
use greeners::CovarianceType;
use ndarray::array;

let asset_returns = array![0.01, 0.02, -0.01, 0.03];
let market_returns = array![0.008, 0.015, -0.005, 0.025];
let risk_free_rate = 0.0001;

let result = CAPM::fit(
    &asset_returns,
    &market_returns,
    risk_free_rate,
    CovarianceType::HC3,
).unwrap();

println!("Beta: {:.4}", result.beta);
println!("Alpha: {:.4}", result.alpha);
println!("R²: {:.4}", result.r_squared);
```

### Fama-French 3 Factor

```rust
use frenchrs::FamaFrench3Factor;
use greeners::CovarianceType;
use ndarray::array;

let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];
let smb = array![0.002, -0.001, 0.003, 0.001, -0.002, 0.001];
let hml = array![0.001, 0.002, -0.002, 0.003, 0.001, -0.001];

let result = FamaFrench3Factor::fit(
    &asset, &market, &smb, &hml,
    0.0001,
    CovarianceType::HC3
).unwrap();

println!("{}", result);
```

### APT (Arbitrage Pricing Theory)

```rust
use frenchrs::APT;
use greeners::CovarianceType;
use ndarray::{array, Array2};

let returns = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005, 0.025];

// Matriz de fatores (n_obs × n_factors)
let factors = Array2::from_shape_vec((7, 3), vec![
    0.008, 0.002, 0.001,
    0.015, -0.001, 0.002,
    -0.005, 0.003, -0.002,
    0.025, 0.001, 0.003,
    0.012, -0.002, 0.001,
    -0.003, 0.001, -0.001,
    0.020, 0.002, 0.002,
]).unwrap();

let factor_names = Some(vec![
    "Market".to_string(),
    "Size".to_string(),
    "Value".to_string(),
]);

let result = APT::fit(
    &returns,
    &factors,
    0.0001,
    CovarianceType::HC3,
    factor_names,
).unwrap();

println!("{}", result);
```

### IVOL & Tracking Error

```rust
use frenchrs::{CAPM, IVOLAnalysis, TrackingErrorAnalysis};
use greeners::CovarianceType;
use ndarray::array;

let asset = array![0.01, 0.02, -0.01, 0.03, 0.015, -0.005];
let market = array![0.008, 0.015, -0.005, 0.025, 0.012, -0.003];

// Estimar CAPM
let capm = CAPM::fit(&asset, &market, 0.0001, CovarianceType::HC3).unwrap();

// Análise de IVOL (Idiosyncratic Volatility)
let ivol = IVOLAnalysis::from_residuals(&capm.residuals).unwrap();
println!("IVOL: {:.4}%", ivol.ivol * 100.0);
println!("IVOL Anualizado: {:.2}%", ivol.ivol_annualized_monthly * 100.0);
println!("Classificação: {}", ivol.ivol_classification());

// Análise de Tracking Error
let te = TrackingErrorAnalysis::new(
    &asset,
    &capm.fitted_values,
    capm.alpha,
    capm.r_squared,
).unwrap();

println!("Tracking Error: {:.4}%", te.tracking_error * 100.0);
println!("Information Ratio: {:.4}", te.information_ratio);
println!("Classificação: {}", te.te_classification());
```

### Rolling Betas

```rust
use frenchrs::RollingCAPM;
use greeners::CovarianceType;
use ndarray::array;

// Dados de 24 meses
let asset = array![/* 24 retornos mensais */];
let market = array![/* 24 retornos mensais */];

// Rolling window de 12 meses
let rolling = RollingCAPM::fit(
    &asset,
    &market,
    0.0025, // taxa livre de risco mensal
    12,     // janela de 12 meses
    CovarianceType::HC3
).unwrap();

// Análise de estabilidade do beta
let stability = rolling.beta_stability();
println!("Beta Médio: {:.4}", stability.mean);
println!("Coef. Variação: {:.4}", stability.coefficient_of_variation);
println!("Classificação: {}", stability.stability_classification());
println!("Tendência: {}", stability.trend_classification());

// Verificar se beta é estável (CV < 10%)
if rolling.is_beta_stable(0.1) {
    println!("Beta estável ao longo do tempo");
}
```

## 📖 Exemplos

Execute os exemplos incluídos:

```bash
# Comparação completa de todos os modelos
cargo run --example complete_comparison

# Demonstração do APT com múltiplos fatores
cargo run --example apt_example

# Análise de risco: IVOL & Tracking Error
cargo run --example risk_analysis

# Rolling Betas: Análise temporal de estabilidade
cargo run --example rolling_betas

# Comparação CAPM vs FF3 vs Carhart vs FF5
cargo run --example model_comparison

# Uso básico do CAPM
cargo run --example capm_example

# Uso com DataFrame
cargo run --example capm_dataframe
```

## 📊 Tipos de Covariância Suportados

- `NonRobust` - OLS clássico (Gauss-Markov)
- `HC0`, `HC1`, `HC2`, `HC3`, `HC4` - Heteroskedasticity-consistent (White)
- `NeweyWest` - Autocorrelation and heteroskedasticity consistent
- `Clustering` - Cluster-robust standard errors

## 🔬 Estatísticas Fornecidas

Todos os modelos fornecem:

- **Parâmetros**: α (alpha), β (betas dos fatores)
- **Inferência**: Erros padrão, estatísticas t, p-values, intervalos de confiança
- **Qualidade de Ajuste**: R², R² ajustado, tracking error, information ratio
- **Diagnóstico**: Resíduos, valores ajustados
- **Classificações**: Performance, tamanho, valor, rentabilidade, etc.

## 🧪 Testes

```bash
# Rodar todos os testes
cargo test --all

# Rodar testes com output
cargo test --all -- --nocapture

# Rodar testes específicos
cargo test --test capm_tests
```

**Cobertura de Testes:**
- 20 testes CAPM
- 17 testes Fama-French 3 Factor
- 10 testes Carhart 4 Factor
- 11 testes Fama-French 6 Factor
- 12 testes APT
- 19 testes IVOL & Tracking Error
- 20 testes Rolling Betas
- 19 testes internos adicionais
- **Total: 128+ testes**

## 📈 Performance

Frenchrs é construído para performance máxima:

- Usa BLAS/LAPACK via `ndarray-linalg` para álgebra linear otimizada
- Aproveitamento de múltiplos núcleos quando disponível
- Zero-copy sempre que possível
- Compilação otimizada com LTO

```toml
[profile.release]
opt-level = 3
lto = true
```

## 🗺️ Roadmap

- [ ] Value-at-Risk (VaR)
- [ ] Conditional VaR (CVaR)
- [ ] Portfolio Optimization (Markowitz, Black-Litterman)
- [ ] Rolling window analysis
- [ ] Bindings para Python (PyO3)
- [ ] Suporte para séries temporais irregulares

## 📚 Referências

### Artigos Fundamentais

1. **Sharpe, W. F.** (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk". *Journal of Finance*, 19(3), 425-442.

2. **Fama, E. F., & French, K. R.** (1993). "Common Risk Factors in the Returns on Stocks and Bonds". *Journal of Financial Economics*, 33(1), 3-56.

3. **Carhart, M. M.** (1997). "On Persistence in Mutual Fund Performance". *Journal of Finance*, 52(1), 57-82.

4. **Fama, E. F., & French, K. R.** (2015). "A Five-Factor Asset Pricing Model". *Journal of Financial Economics*, 116(1), 1-22.

5. **Ross, S. A.** (1976). "The Arbitrage Theory of Capital Asset Pricing". *Journal of Economic Theory*, 13(3), 341-360.

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## 🙏 Agradecimentos

- **Greeners**: Infraestrutura econométrica robusta
- **ndarray**: Arrays N-dimensionais de alto desempenho
- **statrs**: Distribuições estatísticas
- Comunidade Rust de finanças quantitativas

## 📞 Contato

Para questões, sugestões ou bugs, por favor abra uma issue no GitHub.

---

**Desenvolvido com ❤️ em Rust para a comunidade de finanças quantitativas**
