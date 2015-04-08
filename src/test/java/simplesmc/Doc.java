package simplesmc;


import tutorialj.Tutorial;

public class Doc
{
  /**
   * 
   * Summary [![Build Status](https://travis-ci.org/alexandrebouchard/simplesmc.png?branch=master)](https://travis-ci.org/alexandrebouchard/simplesmc)
   * -------
   * 
   * simplesmc is ...
   * 
   * simplesmc stands for ...
   * 
   * 
   * Installation
   * ------------
   * 
   * There are several options available to install the package:
   * 
   * ### Integrate to a gradle script
   * 
   * Simply add the following lines (replacing 1.0.0 by the current version (see git tags)):
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
   *   compile group: 'ca.ubc.stat', name: 'simplesmc', version: '1.0.0'
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
}
