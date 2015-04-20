package simplesmc;

import tutorialj.Tutorial;

public class Doc
{
  /**
   * 
   * Summary
   * -------
   * 
   * An easy to use library for SMC algorithms (Sequential Monte Carlo, AKA particle filters).
   * 
   * The main features are:
   * 
   * 1. Parallelized implementation of SMC with adaptive resampling, including efficient implementations of multinomial, stratified, and systematic resampling.
   * 2. Flexible interfaces for designing proposals, including generic-type particles.
   * 3. Basic PMCMC integration into the probabilistic programming language [Blang](https://github.com/alexandrebouchard/bayonet) for declarative inference of static parameters.
   * 
   * 
   * Installation
   * ------------
   * 
   * There are several ways to install:
   * 
   * ### Integrate to a gradle script
   * 
   * Simply add the following lines (replacing 1.0.2 by the current version (see git tags)):
   * 
   * ```groovy
   * repositories {
   *  mavenCentral()
   *  jcenter()
   *  maven {
   *     url "http://www.stat.ubc.ca/~bouchard/maven/"
   *   }
   * }
   * 
   * dependencies {
   *   compile group: 'ca.ubc.stat', name: 'simplesmc', version: '1.0.2'
   * }
   * ```
   * 
   * ### Compile using the provided gradle script
   * 
   * - Check out the source ``git clone git@github.com:alexandrebouchard/simplesmc.git``
   * - Compile using ``gradle installApp``
   * - Add the jars in ``build/install/simplesmc/lib/`` into your classpath
   * 
   * ### Use in eclipse
   * 
   * - Check out the source ``git clone git@github.com:alexandrebouchard/simplesmc.git``
   * - Type ``gradle eclipse`` from the root of the repository
   * - From eclipse:
   *   - ``Import`` in ``File`` menu
   *   - ``Import existing projects into workspace``
   *   - Select the root
   *   - Deselect ``Copy projects into workspace`` to avoid having duplicates
   */
  @Tutorial(startTutorial = "README.md", showSource = false)
  public void installInstructions()
  {
  }
  
  /**
   * Simple SMC example
   * ------------------
   */
  @Tutorial(showSource = false, nextStep = TestSMC.class)
  public void smcExample() {}

  /**
   * Simple PMCMC example
   * ------------------
   */
  @Tutorial(showSource = false, nextStep = TestPMCMC.class)
  public void pmcmcExample() {}
  
}
